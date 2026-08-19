# SPDX-License-Identifier: Apache-2.0
"""Optional Triton implementation of rectangular Sol-Attn.

The reference implementation in Sana assumes ``q`` and ``k`` have the same
sequence length.  Hunyuan Image uses a cached prefix, so this module keeps
separate query and key block counts and retains the prefix blocks as exact
``sink`` blocks.  Triton is imported lazily by :func:`sol_attn`; importing
vLLM-Omni therefore remains safe on Ascend and CPU installations.

The kernel is written against vLLM's ``triton_utils`` shim rather than
importing CUDA Triton directly.  On an Ascend build this shim resolves to the
Ascend Triton runtime; on CUDA it resolves to upstream Triton.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Triton is an optional accelerator dependency.
    from vllm.triton_utils import tl, triton
except Exception:  # pragma: no cover - environment dependent
    triton = None
    tl = None


BLOCK_SIZE = 64
GROUP_SIZE = 32
HEAD_DIM = 128


def is_available() -> bool:
    """Return whether the optional Triton runtime can be imported."""

    return triton is not None


def _prepare_summaries(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    tau: float,
    scale: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pooled K/V and per-query-block routing thresholds.

    This is deliberately expressed with regular accelerator tensor operations.  The
    expensive token-level attention and online softmax are fused in Triton;
    keeping preprocessing in PyTorch also makes the rectangular adaptation
    easy to audit against the torch.compile reference path.
    """

    batch, q_tokens, heads, dim = query.shape
    k_tokens = key.shape[1]
    q_blocks = (q_tokens + block_size - 1) // block_size
    k_blocks = (k_tokens + block_size - 1) // block_size
    q_padded = q_blocks * block_size
    k_padded = k_blocks * block_size

    q_pad = F.pad(query, (0, 0, 0, 0, 0, q_padded - q_tokens))
    k_pad = F.pad(key, (0, 0, 0, 0, 0, k_padded - k_tokens))
    v_pad = F.pad(value, (0, 0, 0, 0, 0, k_padded - k_tokens))
    q_blocks_view = q_pad.view(batch, q_blocks, block_size, heads, dim)
    k_blocks_view = k_pad.view(batch, k_blocks, block_size, heads, dim)
    v_blocks_view = v_pad.view(batch, k_blocks, block_size, heads, dim)

    q_len = (q_tokens - torch.arange(q_blocks, device=query.device) * block_size).clamp(
        min=0, max=block_size
    ).to(torch.float32)
    k_len = (k_tokens - torch.arange(k_blocks, device=key.device) * block_size).clamp(
        min=0, max=block_size
    ).to(torch.float32)

    q_bar = q_blocks_view.float().sum(dim=2) / q_len[None, :, None, None]
    kc = k_blocks_view.float().sum(dim=2) / k_len[None, :, None, None]
    vc = v_blocks_view.float().sum(dim=2)

    proxy = torch.einsum("bqhd,bkhd->bqhk", q_bar, kc) * scale
    mean = proxy.mean(dim=-1)
    variance = (proxy - mean[..., None]).square().mean(dim=-1)
    # The Triton online softmax uses exp2, so keep routing thresholds in the
    # same log2 domain as the kernel's scaled dot products.
    threshold = (mean + float(tau) * torch.sqrt(variance + 1.0e-6)) * 1.4426950408889634

    # The kernel walks groups of 32 summaries and intentionally loads the
    # final group without a dynamic shape.  Pad summaries with zeros; invalid
    # blocks are excluded by the ``block_indices < NK`` predicate in the
    # kernel.
    summary_blocks = (k_blocks + GROUP_SIZE - 1) // GROUP_SIZE * GROUP_SIZE
    kc_out = torch.zeros((batch, summary_blocks, heads, dim), device=key.device, dtype=key.dtype)
    vc_out = torch.zeros_like(kc_out)
    kc_out[:, :k_blocks] = kc.to(key.dtype)
    vc_out[:, :k_blocks] = vc.to(value.dtype)
    return kc_out, vc_out, threshold.to(torch.float32)


if triton is not None:

    @triton.jit
    def _forward_rect(
        q_ptr,
        k_ptr,
        v_ptr,
        kc_ptr,
        vc_ptr,
        threshold_ptr,
        out_ptr,
        scale,
        TQ,
        TK,
        QSTRIDE,
        NPAD,
        query_offset,
        prefix_len,
        HAS_SINK: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NQ: tl.constexpr,
        NK: tl.constexpr,
        BV: tl.constexpr,
        BLOCK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        v_tile, q_block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        batch, head = batch_head // H, batch_head % H
        offsets = tl.arange(0, BLOCK)
        dims = tl.arange(0, D)
        value_dims = v_tile * BV + tl.arange(0, BV)
        q_tokens = q_block * BLOCK + offsets
        q_valid = q_tokens < TQ
        q_offsets = ((batch * QSTRIDE + q_tokens[:, None]) * H + head) * D + dims[None, :]
        q = tl.load(
            q_ptr + q_offsets,
            mask=tl.broadcast_to(q_valid[:, None], (BLOCK, D)),
            other=0.0,
        )
        q_len = tl.minimum(BLOCK, TQ - q_block * BLOCK).to(tl.float32)

        output = tl.zeros((BLOCK, BV), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK,), dtype=tl.float32)
        # Start at zero instead of -inf.  This keeps the no-approximation
        # path finite and avoids scalar-condition tl.where lowering, which is
        # currently unsupported by Triton-Ascend's BlockPtrAnalysis.
        row_max = tl.zeros((BLOCK,), dtype=tl.float32)
        scale_log2 = scale * 1.4426950408889634
        route_threshold = tl.load(threshold_ptr + (batch * NQ + q_block) * H + head)

        group_offsets = tl.arange(0, GROUP)
        for group_start in range(0, NK, GROUP):
            block_indices = group_start + group_offsets
            valid = block_indices < NK
            kc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + dims[None, :]
            vc_offsets = ((batch * NPAD + block_indices[:, None]) * H + head) * D + value_dims[None, :]
            kc = tl.load(
                kc_ptr + kc_offsets,
                mask=tl.broadcast_to(valid[:, None], (GROUP, D)),
                other=0.0,
            )
            vc = tl.load(
                vc_ptr + vc_offsets,
                mask=tl.broadcast_to(valid[:, None], (GROUP, BV)),
                other=0.0,
            )
            scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
            q_global_start = query_offset + q_block * BLOCK
            k_starts = block_indices * BLOCK
            exact = ((tl.sum(scores, axis=0) / q_len > route_threshold)
                     | (tl.abs(q_global_start - k_starts) <= BLOCK))
            if HAS_SINK:
                exact = exact | (k_starts < prefix_len)
            exact = exact & valid

            approximate = valid & ~exact
            approximate_mask = approximate[None, :].to(tl.float32)
            # Use a finite sentinel and arithmetic masks instead of
            # broadcasting tl.where over [BLOCK, GROUP].  Triton-Ascend's
            # current BlockPtrAnalysis rejects that select shape.
            safe_scores = scores * approximate_mask + (-1.0e9) * (1.0 - approximate_mask)
            new_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.math.exp2(safe_scores - new_max[:, None]) * approximate_mask
            output = output * alpha[:, None] + tl.dot(probability.to(vc.dtype), vc)
            lengths = tl.minimum(BLOCK, tl.maximum(0, TK - k_starts)).to(tl.float32)
            row_sum = row_sum * alpha + tl.sum(probability * lengths[None, :], axis=1)
            row_max = new_max

            exact_offsets = exact.to(tl.int32) * group_offsets + (~exact).to(tl.int32) * GROUP
            num_exact = tl.sum(exact.to(tl.int32), axis=0)
            for _ in range(num_exact):
                offset = tl.min(exact_offsets)
                block = group_start + offset
                replaced = (group_offsets == offset).to(tl.int32)
                exact_offsets = replaced * GROUP + (1 - replaced) * exact_offsets
                kv_tokens = block * BLOCK + offsets
                kv_valid = kv_tokens < TK
                k_offsets = ((batch * TK + kv_tokens[:, None]) * H + head) * D + dims[None, :]
                k = tl.load(
                    k_ptr + k_offsets,
                    mask=tl.broadcast_to(kv_valid[:, None], (BLOCK, D)),
                    other=0.0,
                )
                exact_scores = tl.dot(q, k.T).to(tl.float32) * scale_log2
                valid_mask = kv_valid[None, :].to(tl.float32)
                exact_scores = exact_scores * valid_mask + (-1.0e9) * (1.0 - valid_mask)
                new_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
                alpha = tl.math.exp2(row_max - new_max)
                exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
                row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
                v_offsets = ((batch * TK + kv_tokens[:, None]) * H + head) * D + value_dims[None, :]
                v = tl.load(
                    v_ptr + v_offsets,
                    mask=tl.broadcast_to(kv_valid[:, None], (BLOCK, BV)),
                    other=0.0,
                )
                output = output * alpha[:, None] + tl.dot(exact_probability.to(v.dtype), v)
                row_max = new_max

        out_offsets = ((batch * QSTRIDE + q_tokens[:, None]) * H + head) * D + value_dims[None, :]
        tl.store(
            out_ptr + out_offsets,
            (output / row_sum[:, None]).to(tl.bfloat16),
            mask=tl.broadcast_to(q_valid[:, None], (BLOCK, BV)),
        )


def sol_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    tau: float = 1.0,
    prefix_len: int = 0,
    query_offset: int = 0,
) -> torch.Tensor:
    """Run rectangular Triton Sol-Attn for BF16 BTHD tensors."""

    if triton is None:
        raise RuntimeError("Triton is not installed; use the torch.compile Sol-Attn path")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("query, key, and value must be BTHD with matching K/V shapes")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query and key must match in batch, heads, and head dimension")
    if query.shape[-1] != HEAD_DIM or query.dtype != torch.bfloat16:
        raise ValueError("Triton Sol-Attn requires contiguous BF16 head_dim=128 tensors")
    if key.device != query.device or value.device != query.device:
        raise ValueError("Triton Sol-Attn requires tensors on the same device")
    # Hunyuan's Q/K/V are commonly produced by transpose/reshape views.  The
    # pointer kernel uses the packed BTHD address formula, so materialize only
    # non-contiguous inputs here; the normal contiguous path remains zero-copy.
    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()

    batch, q_tokens, heads, dim = query.shape
    k_tokens = key.shape[1]
    q_blocks = (q_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_blocks = (k_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    kc, vc, threshold = _prepare_summaries(
        query, key, value, tau=tau, scale=scale, block_size=BLOCK_SIZE
    )
    output = torch.empty_like(query)
    _forward_rect[(1, q_blocks, batch * heads)](
        query,
        key,
        value,
        kc,
        vc,
        threshold,
        output,
        float(scale),
        q_tokens,
        k_tokens,
        q_tokens,
        kc.shape[1],
        int(query_offset),
        int(prefix_len),
        HAS_SINK=prefix_len > 0,
        H=heads,
        D=dim,
        NQ=q_blocks,
        NK=k_blocks,
        BV=dim,
        BLOCK=BLOCK_SIZE,
        GROUP=GROUP_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return output


__all__ = ["is_available", "sol_attn"]
