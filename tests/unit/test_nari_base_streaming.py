"""Contract tests for the vendored Nari Base-model streaming extensions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

VENDOR_SRC = Path(__file__).parents[2] / "vendor" / "nari-qwen3-tts" / "src"
sys.path.insert(0, str(VENDOR_SRC))

from nari_qwen3_tts.contract.request import (  # noqa: E402
  EncodedText,
  SynthesisRequest,
  VoiceClonePrompt,
)
from nari_qwen3_tts.engine.codec_readiness import next_codec_readiness  # noqa: E402
from nari_qwen3_tts.engine.state import CodecLane  # noqa: E402
from nari_qwen3_tts.executor.base_prompt_cache import tensor_cache_identity  # noqa: E402
from nari_qwen3_tts.executor.talker import TalkerExecutor  # noqa: E402


class _FakeTalker(nn.Module):
  """Expose only embeddings required by Base prefill preparation."""

  def __init__(self, *, hidden_size: int) -> None:
    """Create deterministic text and codec embedding tables."""
    super().__init__()
    self.text = nn.Embedding(128, hidden_size)
    self.codec = nn.Embedding(128, hidden_size)

  def get_projected_text_embedding_cache(self) -> torch.Tensor:
    """Return the fake projected text table."""
    return self.text.weight

  def get_input_embeddings(self) -> nn.Embedding:
    """Return the fake layer-zero Codec table."""
    return self.codec


def _base_executor(hidden_size: int = 8) -> TalkerExecutor:
  """Construct the Base prefill portion of an executor without CUDA capture."""
  executor = object.__new__(TalkerExecutor)
  executor.model = _FakeTalker(hidden_size=hidden_size)
  executor.device = torch.device("cpu")
  executor.dtype = torch.float32
  executor.reference_codec_embeddings = nn.ModuleList(
    nn.Embedding(128, hidden_size) for _ in range(15)
  )
  predictor = SimpleNamespace(vocab_size=128)
  talker = SimpleNamespace(
    codec_nothink_id=1,
    codec_think_bos_id=2,
    codec_think_eos_id=3,
    codec_think_id=4,
    codec_pad_id=5,
    codec_bos_id=6,
    codec_language_id={"english": 7},
    code_predictor=predictor,
  )
  executor.config = SimpleNamespace(
    talker=talker,
    tts_bos_token_id=8,
    tts_eos_token_id=9,
    tts_pad_token_id=10,
  )
  return executor


def test_base_icl_prefill_reuses_and_invalidates_prompt_embeddings() -> None:
  """Reuse invariant prompt work and discard it after model-weight changes."""
  prompt = VoiceClonePrompt(
    ref_code=torch.zeros((4, 16), dtype=torch.long),
    ref_spk_embedding=torch.ones(8),
    ref_text="reference words",
  )
  request = SynthesisRequest(
    text="target words",
    language="english",
    voice_clone_prompt=prompt,
  )
  prepared = EncodedText(
    request=request,
    text_token_ids=torch.arange(12, dtype=torch.long),
    instruct_token_ids=torch.empty(0, dtype=torch.long),
    ref_token_ids=torch.arange(7, dtype=torch.long),
  )
  executor = _base_executor()

  first = executor._prepare_base_input_plan(prepared)
  second = executor._prepare_base_input_plan(prepared)

  assert first.embedding_overrides is not None
  assert second.embedding_overrides is not None
  assert torch.equal(first.embedding_overrides, second.embedding_overrides)
  assert executor.base_prompt_cache_stats().hits == 1
  assert executor.base_prompt_cache_stats().misses == 1
  executor.clear_base_prompt_cache()
  executor._prepare_base_input_plan(prepared)
  stats = executor.base_prompt_cache_stats()
  assert stats.entries == 1
  assert stats.hits == 1
  assert stats.misses == 2


def test_inference_tensor_cache_identity_uses_stable_fallback() -> None:
  """Accept prompt tensors created by inference-mode tokenizer execution."""
  with torch.inference_mode():
    prompt = torch.arange(8)

  pointer, version, shape = tensor_cache_identity(prompt)

  assert pointer == prompt.data_ptr()
  assert version == 0
  assert shape == (8,)


def test_reference_codec_prefix_is_decoded_but_not_visible() -> None:
  """Verify stateful Codec context hides only reference frames from PCM output."""
  lane = CodecLane(
    chunk_schedule=(4,),
    buffered_frames=tuple(torch.zeros(16) for _ in range(4)),
    history_frames=tuple(torch.zeros(16) for _ in range(4)),
    hidden_prefix_frames=3,
  )

  readiness = next_codec_readiness(lane)

  assert readiness is not None
  assert readiness.compatibility is not None
  assert readiness.compatibility.input_frames == 4
  assert readiness.compatibility.pcm_start_frame == 3
  assert readiness.compatibility.visible_frames == 1
