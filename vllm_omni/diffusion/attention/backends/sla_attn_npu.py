# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton and AscendC kernels for rectangular SLA attention.

Adapted from ``FastVideo_fork/fastvideo/attention/backends/sla_npu.py``.
Only the inference forward path is retained. Unlike the FastVideo source,
query and KV sequence lengths have independent strides so cached-prefix
Hunyuan Image 3 attention is supported.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch_npu
from triton.runtime import driver
from vllm.triton_utils import tl, triton


def _get_npu_ai_core_count() -> int:
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    return int(properties["num_aicore"])


@triton.jit
def _compress_kernel(
    x,
    mean_x,
    length: tl.constexpr,
    dim: tl.constexpr,
    block_length: tl.constexpr,
):
    idx_bh = tl.program_id(0)
    nproc = tl.program_id(1)

    compressed_length = (length + block_length - 1) // block_length
    for compressed_idx in tl.range(nproc, compressed_length, tl.num_programs(1)):
        token_start = compressed_idx * block_length
        x_start = x + idx_bh * length * dim + token_start * dim
        mean_start = mean_x + idx_bh * compressed_length * dim + compressed_idx * dim

        x_offsets = tl.arange(0, block_length)[:, None] * dim + tl.arange(0, dim)[None, :]
        x_mask = token_start + tl.arange(0, block_length)[:, None] < length
        values = tl.load(x_start + x_offsets, mask=x_mask)
        valid_length = min(block_length, length - token_start)
        mean = tl.sum(values, axis=0) / valid_length
        tl.store(mean_start + tl.arange(0, dim), mean)


def mean_pool(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if not x.is_contiguous():
        raise ValueError("SLA mean_pool requires a contiguous BNSD tensor")
    batch, heads, length, dim = x.shape
    length_blocks = (length + block_size - 1) // block_size
    pooled = torch.empty((batch, heads, length_blocks, dim), device=x.device, dtype=x.dtype)
    bh_blocks = batch * heads
    worker_blocks = max(1, min(length_blocks, 32768 // max(1, bh_blocks)))
    _compress_kernel[(bh_blocks, worker_blocks)](x, pooled, length, dim, block_size)
    return pooled


def get_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    blkq: int,
    blkk: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build an SLA top-k map with independent query and key block counts."""

    centered_k = k - torch.mean(k, dim=-2, keepdim=True)
    pooled_q = mean_pool(q, blkq)
    pooled_k = mean_pool(centered_k.contiguous(), blkk)
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)

    key_blocks = pooled_score.shape[-1]
    topk = max(1, min(key_blocks, int(topk_ratio * key_blocks)))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices.contiguous()
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map.contiguous(), lut, topk


@triton.jit
def _attn_fwd(
    query,
    key,
    value,
    output,
    lut,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    batch: tl.constexpr,
    heads: tl.constexpr,
    lq: tl.constexpr,
    lk: tl.constexpr,
    dim: tl.constexpr,
    physical_block_m: tl.constexpr,
    logical_block_m: tl.constexpr,
    block_n: tl.constexpr,
    logical_m_blocks: tl.constexpr,
    num_cores: tl.constexpr,
):
    m_factor = logical_block_m // physical_block_m
    physical_m_blocks = triton.cdiv(lq, physical_block_m)
    total_blocks = physical_m_blocks * batch * heads
    pid = tl.program_id(0)

    for block_idx in range(pid, total_blocks, num_cores):
        bh_idx = block_idx // physical_m_blocks
        physical_m_idx = block_idx % physical_m_blocks
        logical_m_idx = physical_m_idx // m_factor

        q_offset = bh_idx.to(tl.int64) * lq * dim
        kv_offset = bh_idx.to(tl.int64) * lk * dim
        lut_offset = (bh_idx.to(tl.int64) * logical_m_blocks + logical_m_idx.to(tl.int64)) * topk

        offsets_m = physical_m_idx * physical_block_m + tl.arange(0, physical_block_m)
        offsets_n = tl.arange(0, block_n)
        offsets_d = tl.arange(0, dim)
        query_ptrs = query + q_offset + offsets_m[:, None] * dim + offsets_d[None, :]
        key_ptrs = key + kv_offset + offsets_n[:, None] * dim + offsets_d[None, :]
        value_ptrs = value + kv_offset + offsets_n[:, None] * dim + offsets_d[None, :]
        output_ptrs = output + q_offset + offsets_m[:, None] * dim + offsets_d[None, :]
        lut_ptr = lut + lut_offset

        running_max = tl.full([physical_block_m], -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros([physical_block_m], dtype=tl.float32)
        accumulator = tl.zeros([physical_block_m, dim], dtype=tl.float32)
        q = tl.load(query_ptrs, mask=offsets_m[:, None] < lq)

        for selected_idx in tl.range(topk):
            key_block_idx = tl.load(lut_ptr + selected_idx)
            key_mask = offsets_n < lk - key_block_idx * block_n
            k = tl.load(key_ptrs + key_block_idx * block_n * dim, mask=key_mask[:, None])
            scores = tl.dot(q, tl.trans(k)) * (qk_scale * 1.4426950408889634)
            if lk - key_block_idx * block_n < block_n:
                scores = tl.where(key_mask[None, :], scores, float("-inf"))

            v = tl.load(value_ptrs + key_block_idx * block_n * dim, mask=key_mask[:, None])
            block_max = tl.max(scores, 1)
            new_max = tl.maximum(running_max, block_max)
            probabilities = tl.math.exp2(scores - new_max[:, None])
            correction = tl.math.exp2(running_max - new_max)
            accumulator = accumulator * correction[:, None]
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            running_sum = running_sum * correction + tl.sum(probabilities, 1)
            running_max = new_max

        result = accumulator / running_sum[:, None]
        tl.store(output_ptrs, result.to(output.type.element_ty), mask=offsets_m[:, None] < lq)


def sparse_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    topk: int,
    blkq: int,
    blkk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the rectangular forward-only NPU Triton SLA kernel."""

    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous() or not lut.is_contiguous():
        raise ValueError("NPU Triton SLA_ATTN requires contiguous q, k, v, and lut tensors")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or k.shape != v.shape:
        raise ValueError(f"NPU Triton SLA_ATTN received incompatible shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    if q.shape[-1] not in (64, 128) or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "NPU Triton SLA_ATTN requires matching head_dim 64 or 128, "
            f"got {q.shape[-1]} and {k.shape[-1]}"
        )
    if blkq not in (64, 128) or blkk not in (64, 128):
        raise ValueError(f"NPU Triton SLA_ATTN requires blkq/blkk in (64, 128), got blkq={blkq}, blkk={blkk}")

    output = torch.empty_like(q)
    physical_blkq = 32 if blkq == 128 else blkq
    logical_m_blocks = (q.shape[2] + blkq - 1) // blkq
    num_cores = _get_npu_ai_core_count()
    _attn_fwd[(num_cores,)](
        query=q,
        key=k,
        value=v,
        output=output,
        lut=lut,
        qk_scale=softmax_scale,
        topk=topk,
        batch=q.shape[0],
        heads=q.shape[1],
        lq=q.shape[2],
        lk=k.shape[2],
        dim=q.shape[3],
        physical_block_m=physical_blkq,
        logical_block_m=blkq,
        block_n=blkk,
        logical_m_blocks=logical_m_blocks,
        num_cores=num_cores,
    )
    return output


@lru_cache(maxsize=1)
def _get_ascendc_op():
    try:
        from mindiesd.layers._custom_ops import block_sparse_attention
    except ImportError as exc:
        raise ImportError(
            "AscendC SLA_ATTN requires MindIE-SD with libPTAExtensionOPS.so installed"
        ) from exc
    return block_sparse_attention


def sparse_attention_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_map: torch.Tensor,
    blkq: int,
    blkk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run MindIE-SD's forward-only AscendC block-sparse attention op."""

    if q.shape[-1] not in (64, 128):
        raise ValueError(f"AscendC SLA_ATTN requires head_dim 64 or 128, got {q.shape[-1]}")
    if blkq != 128 or blkk != 128:
        raise ValueError(f"AscendC SLA_ATTN requires blkq=128 and blkk=128, got blkq={blkq}, blkk={blkk}")

    device = torch_npu.npu.current_device()
    device_name = torch_npu.npu.get_device_properties(device).name
    inner_precise = 4 if "950" in device_name else 0
    output, _ = _get_ascendc_op()(
        query=q,
        key=k,
        value=v,
        block_sparse_mask=sparse_map.contiguous(),
        block_shape=[blkq, blkk],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=k.shape[1],
        scale_value=softmax_scale,
        inner_precise=inner_precise,
        softmax_lse_flag=0,
    )
    return output


__all__ = ["get_block_map", "sparse_attention_ascendc", "sparse_attention_triton"]
