# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the torch.compile Sol-Attn reference core."""

import torch
import pytest

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sol_attn import _sol_attn_torch_impl


pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _dense_attention(query, key, value, scale):
    scores = torch.einsum("bqhd,bkhd->bqhk", query, key) * scale
    weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.einsum("bqhk,bkhd->bqhd", weights, value)


def test_sol_attn_all_selected_matches_dense_attention():
    torch.manual_seed(0)
    query = torch.randn(1, 96, 2, 128)
    key = torch.randn(1, 160, 2, 128)
    value = torch.randn_like(key)
    scale = 128**-0.5

    # A sufficiently low threshold routes every K block; capacity then makes
    # the reference path exactly equivalent to regular rectangular attention.
    output, overflow = _sol_attn_torch_impl(query, key, value, scale, -100.0, 64, 3, 64, 64)

    assert not overflow.item()
    torch.testing.assert_close(output, _dense_attention(query, key, value, scale), rtol=2e-5, atol=2e-5)


def test_sol_attn_reports_overflow_instead_of_truncating_routes():
    query = torch.ones(1, 64, 1, 128)
    key = torch.ones(1, 192, 1, 128)
    value = torch.ones_like(key)

    output, overflow = _sol_attn_torch_impl(query, key, value, 128**-0.5, -1.0, 64, 1, 64, 64)

    assert output.shape == query.shape
    assert overflow.item()


def test_sol_attn_is_registered():
    backend = DiffusionAttentionBackendEnum.SOL_ATTN.get_class()
    assert backend.get_name() == "SOL_ATTN"
