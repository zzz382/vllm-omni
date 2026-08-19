# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the NPU SLA backend configuration."""

import pytest
import torch

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sla_attn import SLAAttentionImpl, _resolve_sla_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sla_attn_is_registered():
    backend = DiffusionAttentionBackendEnum.SLA_ATTN.get_class()
    assert backend.get_name() == "SLA_ATTN"


def test_sla_defaults_prefer_ascendc_compatible_blocks():
    assert _resolve_sla_config(None) == (0.95, "auto", 128, 128)


def test_explicit_triton_defaults_to_64_blocks():
    assert _resolve_sla_config({"kernel": "triton", "sparsity": 0.9}) == (0.9, "triton", 64, 64)


def test_auto_with_a_64_block_selects_triton():
    assert _resolve_sla_config({"kernel": "auto", "blkq": 64, "blkk": 128}) == (0.95, "triton", 64, 128)


def test_gqa_kv_heads_are_expanded_for_the_sparse_kernel():
    query = torch.empty(1, 8, 4, 16)
    key = torch.arange(2.0).reshape(1, 1, 2, 1).expand(1, 8, 2, 16)
    value = key + 10
    expanded_key, expanded_value = SLAAttentionImpl._expand_kv_heads(query, key, value)

    assert expanded_key.shape == query.shape
    assert expanded_key[0, 0, :, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert expanded_value[0, 0, :, 0].tolist() == [10.0, 10.0, 11.0, 11.0]


@pytest.mark.parametrize("blkq, blkk", [(64, 128), (128, 64), (64, 64)])
def test_ascendc_requires_128_blocks(blkq, blkk):
    with pytest.raises(ValueError, match="requires blkq=128 and blkk=128"):
        _resolve_sla_config({"kernel": "ascendc", "blkq": blkq, "blkk": blkk})


@pytest.mark.parametrize("blkq, blkk", [(32, 64), (64, 256), (0, 128)])
def test_triton_rejects_unsupported_blocks(blkq, blkk):
    with pytest.raises(ValueError, match="requires blkq and blkk to be 64 or 128"):
        _resolve_sla_config({"kernel": "triton", "blkq": blkq, "blkk": blkk})


@pytest.mark.parametrize("sparsity", [-0.1, 1.0, float("inf"), float("nan")])
def test_sparsity_range_is_validated(sparsity):
    with pytest.raises(ValueError, match="0 <= sparsity < 1"):
        _resolve_sla_config({"sparsity": sparsity})
