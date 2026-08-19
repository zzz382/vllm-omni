# SPDX-License-Identifier: Apache-2.0
"""NPU SLA block-sparse attention for Hunyuan Image 3.

The routing and sparse kernels are adapted from FastVideo's NPU SLA backend.
Hunyuan Image 3 reuses a prompt KV prefix after the first denoising step, so
the accelerator implementation supports rectangular attention (``LQ != LK``).

This inference backend intentionally implements the SLA sparse branch only.
The trainable linear compensation branch from the SLA paper requires model
specific ``proj_l`` weights, which the original Hunyuan Image 3 checkpoint
does not contain.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

logger = init_logger(__name__)

_SUPPORTED_BLOCK_SIZES = frozenset({64, 128})
_SUPPORTED_HEAD_SIZES = [64, 128]
_DEFAULT_SPARSITY = 0.95
_AUTO_ASCENDC_DISABLED = False


def _resolve_sla_config(backend_kwargs: dict[str, Any] | None) -> tuple[float, str, int, int]:
    """Validate service configuration and resolve the effective kernel route."""

    kwargs = backend_kwargs or {}
    sparsity = float(kwargs.get("sparsity", _DEFAULT_SPARSITY))
    if not math.isfinite(sparsity) or not 0.0 <= sparsity < 1.0:
        raise ValueError(f"SLA_ATTN sparsity must satisfy 0 <= sparsity < 1, got {sparsity}")

    requested_kernel = str(kwargs.get("kernel", "auto")).lower()
    if requested_kernel not in {"auto", "ascendc", "triton"}:
        raise ValueError("SLA_ATTN kernel must be one of: auto, ascendc, triton")

    default_block = 64 if requested_kernel == "triton" else 128
    raw_blkq = kwargs.get("blkq")
    raw_blkk = kwargs.get("blkk")
    blkq = default_block if raw_blkq is None else int(raw_blkq)
    blkk = default_block if raw_blkk is None else int(raw_blkk)

    if requested_kernel == "ascendc":
        if blkq != 128 or blkk != 128:
            raise ValueError(f"AscendC SLA_ATTN requires blkq=128 and blkk=128, got blkq={blkq}, blkk={blkk}")
        return sparsity, "ascendc", blkq, blkk

    if blkq not in _SUPPORTED_BLOCK_SIZES or blkk not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(
            "NPU Triton SLA_ATTN requires blkq and blkk to be 64 or 128, "
            f"got blkq={blkq}, blkk={blkk}"
        )

    # auto with 128x128 prefers AscendC and falls back to Triton. A 64 block
    # explicitly selects Triton because the AscendC operator only accepts 128.
    effective_kernel = "auto" if requested_kernel == "auto" and blkq == blkk == 128 else "triton"
    return sparsity, effective_kernel, blkq, blkk


class SLAAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    supported_platforms = ("npu",)

    @classmethod
    def supports_attention_mask(cls) -> bool:
        return False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return _SUPPORTED_HEAD_SIZES

    @staticmethod
    def get_name() -> str:
        return "SLA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SLAAttentionImpl"]:
        return SLAAttentionImpl


class SLAAttentionImpl(AttentionImpl):
    """Forward-only SLA block-sparse attention for Ascend NPU."""

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
        del extra_impl_args
        if head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"SLA_ATTN supports head_size 64 or 128, got {head_size}")
        if causal:
            raise ValueError("SLA_ATTN is non-causal; use FLASH_ATTN for causal attention")
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = float(softmax_scale)
        self.qkv_layout = qkv_layout
        self.sparsity, self.kernel, self.blkq, self.blkk = _resolve_sla_config(backend_kwargs)

        self.dense_fallback = FlashAttentionBackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            qkv_layout=qkv_layout,
        )

    @staticmethod
    def _sparse_enabled(attn_metadata: AttentionMetadata | None) -> bool:
        if attn_metadata is None:
            return False
        extra = attn_metadata.extra
        return bool(extra.get("sparse_attn_enabled", extra.get("sol_attn_enabled", False)))

    def _can_run_sparse(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
        return (
            query.ndim == 4
            and key.ndim == 4
            and value.shape == key.shape
            and query.shape[0] == key.shape[0]
            and key.shape[2] > 0
            and query.shape[2] % key.shape[2] == 0
            and query.shape[3] == key.shape[3]
            and query.shape[-1] == self.head_size
            and query.shape[1] > 0
            and key.shape[1] > 0
        )

    @staticmethod
    def _expand_kv_heads(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeat = query.shape[2] // key.shape[2]
        if repeat == 1:
            return key, value
        return key.repeat_interleave(repeat, dim=2), value.repeat_interleave(repeat, dim=2)

    def _forward_triton(self, q, k, v, lut, topk):
        from vllm_omni.diffusion.attention.backends.sla_attn_npu import sparse_attention_triton

        return sparse_attention_triton(
            q,
            k,
            v,
            lut,
            topk,
            self.blkq,
            self.blkk,
            self.softmax_scale,
        )

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        global _AUTO_ASCENDC_DISABLED

        if not self._sparse_enabled(attn_metadata) or not self._can_run_sparse(query, key, value):
            return self.dense_fallback.forward_npu(query, key, value, attn_metadata)

        from vllm_omni.diffusion.attention.backends import sla_attn_npu

        # vLLM diffusion attention uses BSND; SLA kernels use BNSD.
        key, value = self._expand_kv_heads(query, key, value)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        sparse_map, lut, topk = sla_attn_npu.get_block_map(
            q,
            k,
            topk_ratio=1.0 - self.sparsity,
            blkq=self.blkq,
            blkk=self.blkk,
        )

        if self.kernel == "triton":
            output = self._forward_triton(q, k, v, lut, topk)
        elif self.kernel == "ascendc":
            output = sla_attn_npu.sparse_attention_ascendc(
                q,
                k,
                v,
                sparse_map,
                self.blkq,
                self.blkk,
                self.softmax_scale,
            )
        else:
            if _AUTO_ASCENDC_DISABLED:
                output = self._forward_triton(q, k, v, lut, topk)
            else:
                try:
                    output = sla_attn_npu.sparse_attention_ascendc(
                        q,
                        k,
                        v,
                        sparse_map,
                        self.blkq,
                        self.blkk,
                        self.softmax_scale,
                    )
                except (ImportError, FileNotFoundError, RuntimeError) as exc:
                    _AUTO_ASCENDC_DISABLED = True
                    logger.warning("AscendC SLA_ATTN is unavailable; falling back to NPU Triton: %s", exc)
                    output = self._forward_triton(q, k, v, lut, topk)

        return output.transpose(1, 2).contiguous().to(query.dtype)


__all__ = ["SLAAttentionBackend", "SLAAttentionImpl"]
