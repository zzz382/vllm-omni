# SPDX-License-Identifier: Apache-2.0
"""Torch-compiled Sol-Attn reference backend.

This implementation is intentionally expressed with regular PyTorch tensor
operations so it can be lowered by ``torch.compile`` on Ascend NPU.  It is a
correctness-first backend; a fused Ascend kernel can replace the function
without changing the metadata contract.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend


_BLOCK_SIZE = 64
_HEAD_DIM = 128
_MIN_TOKENS = 256
_DEFAULT_MAX_EXACT_BLOCKS = 32


def _pad_tokens(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] == length:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, 0, 0, length - x.shape[1]))


def _sol_attn_torch_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    tau: float,
    block_size: int,
    max_exact_blocks: int,
    prefix_len: int,
    query_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular Sol-Attn with fixed-shape block loops.

    Q/K/V use BSND layout. Unselected blocks use pooled K and summed V;
    selected blocks use original K/V gathered into a fixed-capacity buffer.
    The returned overflow flag requests a dense fallback rather than silently
    truncating a threshold-routed block set.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if max_exact_blocks < 1:
        raise ValueError(f"max_exact_blocks must be at least one, got {max_exact_blocks}")

    batch, q_tokens, heads, dim = query.shape
    k_tokens = key.shape[1]
    q_blocks = (q_tokens + block_size - 1) // block_size
    k_blocks = (k_tokens + block_size - 1) // block_size
    q_padded = q_blocks * block_size
    k_padded = k_blocks * block_size

    q = _pad_tokens(query, q_padded).reshape(batch, q_blocks, block_size, heads, dim)
    k = _pad_tokens(key, k_padded).reshape(batch, k_blocks, block_size, heads, dim)
    v = _pad_tokens(value, k_padded).reshape(batch, k_blocks, block_size, heads, dim)
    k_valid = torch.arange(k_padded, device=key.device) < k_tokens
    q_valid = torch.arange(q_padded, device=query.device) < q_tokens
    block_lengths = k_valid.reshape(k_blocks, block_size).sum(dim=1).to(torch.float32)
    q_lengths = q_valid.reshape(q_blocks, block_size).sum(dim=1).to(torch.float32)

    kc = k.float().sum(dim=2) / block_lengths[None, :, None, None]
    vc = v.float().sum(dim=2)
    q_bar = q.float().sum(dim=2) / q_lengths[None, :, None, None]

    # Proxy statistics are computed per batch/query-block/head.
    proxy = torch.einsum("bqhd,bkhd->bqhk", q_bar, kc) * scale
    proxy_mean = proxy.mean(dim=-1)
    proxy_var = (proxy - proxy_mean[..., None]).square().mean(dim=-1)
    threshold = proxy_mean + tau * torch.sqrt(proxy_var + 1.0e-6)
    route = proxy > threshold[..., None]

    k_indices = torch.arange(k_blocks, device=query.device)
    q_global = query_offset + torch.arange(q_blocks, device=query.device) * block_size
    local = (q_global[:, None] - k_indices[None, :] * block_size).abs() <= block_size
    sink = (k_indices[None, :] * block_size < prefix_len) if prefix_len > 0 else torch.zeros_like(local)
    route = route | local[None, :, None, :] | sink[None, None, None, :]

    max_exact_blocks = min(max_exact_blocks, k_blocks)
    route_scores = torch.where(route, proxy, torch.full_like(proxy, float("-inf")))
    selected_scores, selected_indices = torch.topk(route_scores, max_exact_blocks, dim=-1)
    selected_valid = torch.isfinite(selected_scores)
    overflow = route.sum(dim=-1).amax() > max_exact_blocks
    selected_mask = torch.zeros_like(route).scatter(-1, selected_indices, selected_valid)

    # The proxy branch is dense over pooled K/V, but never touches raw K/V.
    q_rows = q.float().permute(0, 1, 3, 2, 4)
    proxy_scores = torch.einsum("bqhtd,bkhd->bqhtk", q_rows, kc) * scale
    proxy_scores = proxy_scores.masked_fill(selected_mask.unsqueeze(3), float("-inf"))
    m = proxy_scores.amax(dim=-1)
    proxy_p = torch.nan_to_num(torch.exp(proxy_scores - m[..., None]))
    l = (proxy_p * block_lengths[None, None, None, None, :]).sum(dim=-1)
    acc = torch.einsum("bqhtk,bkhd->bqhtd", proxy_p, vc)

    # The selected block dimension is static, so torch.compile can lower this
    # loop without any dynamic nonzero/gather result shape.
    k_bh = k.float().permute(0, 3, 1, 2, 4)
    v_bh = v.float().permute(0, 3, 1, 2, 4)
    valid_blocks = k_valid.reshape(k_blocks, block_size)
    for selected_rank in range(max_exact_blocks):
        index = selected_indices[..., selected_rank]
        valid_selection = selected_valid[..., selected_rank]
        gather_index = index[..., None, None, None].expand(batch, q_blocks, heads, 1, block_size, dim)
        selected_k = torch.gather(k_bh[:, None].expand(batch, q_blocks, heads, k_blocks, block_size, dim), 3, gather_index)
        selected_v = torch.gather(v_bh[:, None].expand(batch, q_blocks, heads, k_blocks, block_size, dim), 3, gather_index)
        selected_k = selected_k.squeeze(3)
        selected_v = selected_v.squeeze(3)
        token_valid = valid_blocks[index] & valid_selection[..., None]
        exact_scores = torch.einsum("bqhtd,bqhsd->bqhts", q_rows, selected_k) * scale
        exact_scores = exact_scores.masked_fill(~token_valid[..., None, :], float("-inf"))
        exact_m = exact_scores.amax(dim=-1)
        new_m = torch.maximum(m, exact_m)
        alpha = torch.nan_to_num(torch.exp(m - new_m))
        exact_p = torch.nan_to_num(torch.exp(exact_scores - new_m[..., None]))
        acc = acc * alpha[..., None] + torch.einsum("bqhts,bqhsd->bqhtd", exact_p, selected_v)
        l = l * alpha + exact_p.sum(dim=-1)
        m = new_m

    out = (acc / l.clamp_min(1.0)[..., None]).permute(0, 1, 3, 2, 4)
    out = out.reshape(batch, q_padded, heads, dim)
    return out[:, :q_tokens].to(query.dtype), overflow


@lru_cache(maxsize=1)
def _compiled_sol_attn():
    return torch.compile(_sol_attn_torch_impl, backend="npugraph_ex", mode="reduce-overhead", dynamic=False)


class SolAttnBackend(AttentionBackend):
    accept_output_buffer = True
    supported_platforms = ("npu",)

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # Sol-Attn is enabled only for the validated rectangular image-query
        # path, whose mask is implicit in the prefix/current-image layout.
        return False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SolAttnImpl"]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl):
    """NPU Sol-Attn using a torch.compile-lowered reference implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        if head_size != _HEAD_DIM:
            raise ValueError(f"SOL_ATTN currently requires head_size={_HEAD_DIM}, got {head_size}")
        if causal:
            raise ValueError("SOL_ATTN is non-causal; use FLASH_ATTN for causal attention")
        self.num_heads = num_heads
        self.softmax_scale = float(softmax_scale)
        self.qkv_layout = qkv_layout
        kwargs = backend_kwargs or {}
        self.tau = float(kwargs.get("tau", 1.0))
        self.block_size = int(kwargs.get("block_size", _BLOCK_SIZE))
        self.max_exact_blocks = int(kwargs.get("max_exact_blocks", _DEFAULT_MAX_EXACT_BLOCKS))
        self.compile_enabled = bool(kwargs.get("compile", True))
        if self.block_size != _BLOCK_SIZE:
            raise ValueError(f"SOL_ATTN currently requires block_size={_BLOCK_SIZE}, got {self.block_size}")
        if self.max_exact_blocks < 1:
            raise ValueError("SOL_ATTN max_exact_blocks must be at least one")
        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    def forward_cuda(self, query, key, value, attn_metadata=None):
        return self.dense_fallback.forward_cuda(query, key, value, attn_metadata)

    def forward_xpu(self, query, key, value, attn_metadata=None):
        return self.dense_fallback.forward_xpu(query, key, value, attn_metadata)

    def forward_npu(self, query, key, value, attn_metadata=None):
        extra = attn_metadata.extra if attn_metadata is not None else {}
        if not extra.get("sol_attn_enabled", False):
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        if query.shape[-1] != _HEAD_DIM or query.shape[2] != key.shape[2]:
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        if query.shape[1] < _MIN_TOKENS:
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
        prefix_len = int(extra.get("sol_attn_prefix_len", max(key.shape[1] - query.shape[1], 0)))
        query_offset = int(extra.get("sol_attn_query_offset", prefix_len))
        try:
            fn = _compiled_sol_attn() if self.compile_enabled else _sol_attn_torch_impl
            output, overflow = fn(
                query,
                key,
                value,
                self.softmax_scale,
                self.tau,
                self.block_size,
                self.max_exact_blocks,
                prefix_len,
                query_offset,
            )
            if bool(overflow.item()):
                return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
            return output
        except Exception:
            # Keep backend selection usable on devices whose current compiler
            # lacks a lowering for one of the reference operators.
            output, overflow = _sol_attn_torch_impl(
                query,
                key,
                value,
                self.softmax_scale,
                self.tau,
                self.block_size,
                self.max_exact_blocks,
                prefix_len,
                query_offset,
            )
            if bool(overflow.item()):
                return self.dense_fallback.forward_npu(query, key, value, attn_metadata)
            return output


__all__ = ["SolAttnBackend", "SolAttnImpl"]
