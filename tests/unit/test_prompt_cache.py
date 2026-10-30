"""Safe PyTorch prompt cache round-trip tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.utils.prompt_cache import load_qwen_prompt, save_qwen_prompt

VoiceClonePromptItem = pytest.importorskip(
  "qwen_tts.inference.qwen3_tts_model",
  reason="PyTorch Qwen prompts require the CPU or CUDA runtime",
).VoiceClonePromptItem


def test_qwen_prompt_uses_json_and_safetensors(tmp_path: Path) -> None:
  """Round-trip every prompt field without creating executable pickle data."""
  path = tmp_path / "prompt"
  original = [
    VoiceClonePromptItem(
      ref_code=torch.tensor([[1, 2], [3, 4]]),
      ref_spk_embedding=torch.tensor([0.25, 0.75]),
      x_vector_only_mode=False,
      icl_mode=True,
      ref_text="Reference text.",
    )
  ]
  save_qwen_prompt(path, original)
  assert path.with_suffix(".json").is_file()
  assert path.with_suffix(".safetensors").is_file()
  assert not list(tmp_path.glob("*.pt"))
  assert not list(tmp_path.glob("*.prompt"))

  restored = load_qwen_prompt(path)
  assert isinstance(restored, list)
  item = restored[0]
  assert item.ref_code is not None
  original_code = original[0].ref_code
  assert original_code is not None
  assert torch.equal(item.ref_code, original_code)
  assert torch.equal(item.ref_spk_embedding, original[0].ref_spk_embedding)
  assert item.icl_mode is True
  assert item.x_vector_only_mode is False
  assert item.ref_text == "Reference text."
