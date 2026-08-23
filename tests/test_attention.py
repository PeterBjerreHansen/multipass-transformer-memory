import warnings

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralAttention


def test_flex_matches_reference_small_cpu():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    flex = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False, flex_block_size=16)
    flex.load_state_dict(ref.state_dict())
    ref.eval(); flex.eval()
    x = torch.randn(2, 17, cfg.hidden_size)
    pos = torch.arange(17)[None, :].expand(2, -1)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yr, _ = ref(x, attention_mask=None, position_ids=pos, use_cache=False)
        yf, _ = flex(x, attention_mask=None, position_ids=pos, use_cache=False)
    torch.testing.assert_close(yf, yr, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("length", [31, 32, 33, 127, 128, 129, 257])
def test_flex_default_block_size_matches_reference(length):
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    flex = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False)
    flex.load_state_dict(ref.state_dict())
    ref.eval(); flex.eval()
    x = torch.randn(1, length, cfg.hidden_size)
    pos = torch.arange(length)[None, :]
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yr, _ = ref(x, attention_mask=None, position_ids=pos, use_cache=False)
        yf, _ = flex(x, attention_mask=None, position_ids=pos, use_cache=False)
    torch.testing.assert_close(yf, yr, atol=2e-5, rtol=2e-5)


def test_padding_forces_correct_reference_fallback():
    cfg = micro_config(sliding_window=4)
    a = MistralAttention(cfg, 0, attention_backend="reference")
    b = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, cfg.hidden_size)
    pos = torch.arange(8)[None, :]
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        ya, _ = a(x, attention_mask=mask, position_ids=pos)
        yb, _ = b(x, attention_mask=mask, position_ids=pos)
    torch.testing.assert_close(ya, yb)


def test_flex_key_validity_mask_matches_reference_when_available():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    flex = MistralAttention(cfg, 0, attention_backend="flex", compile_flex=False, flex_block_size=16)
    flex.load_state_dict(ref.state_dict())
    ref.eval(); flex.eval()
    x = torch.randn(2, 17, cfg.hidden_size)
    pos = torch.arange(17)[None, :].expand(2, -1)
    # Mimics write-only MEM keys: queries remain present, selected K/V positions do not.
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[:, [3, 8, 13]] = False
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yr, _ = ref(x, attention_mask=valid, position_ids=pos, use_cache=False)
        yf, _ = flex(x, attention_mask=valid, position_ids=pos, use_cache=False)
    torch.testing.assert_close(yf, yr, atol=3e-5, rtol=3e-5)


def test_sparse_swa_flex_mask_matches_reference_union():
    cfg = micro_config(sliding_window=4)
    ref = MistralAttention(cfg, 0, attention_backend="reference")
    flex = MistralAttention(
        cfg,
        0,
        attention_backend="flex",
        compile_flex=False,
        flex_block_size=16,
    )
    flex.load_state_dict(ref.state_dict())
    ref.configure_sparse_attention(stride=3, window=2)
    flex.configure_sparse_attention(stride=3, window=2)
    ref.eval()
    flex.eval()
    x = torch.randn(2, 19, cfg.hidden_size)
    pos = torch.arange(19)[None, :].expand(2, -1)
    valid = torch.ones(2, 19, dtype=torch.bool)
    valid[1, [2, 11]] = False
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        expected, _ = ref(x, attention_mask=valid, position_ids=pos)
        actual, _ = flex(x, attention_mask=valid, position_ids=pos)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
