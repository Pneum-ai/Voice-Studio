"""Regression coverage for full-text Base capture budgeting and lossless splitting."""

from __future__ import annotations

import unittest

import torch
from nari_qwen3_tts.contract.request import EncodedText, SynthesisRequest, VoiceClonePrompt

from app.backends.pytorch.nari_input_budget import base_prefill_tokens, split_to_budget
from app.core.errors import ValidationError
from tests.unit.test_nari_base_streaming import _base_executor


class NariInputBudgetTests(unittest.TestCase):
  """Compare budgeting to the real vendored layout without loading model weights."""

  def test_budget_matches_executor_layout(self) -> None:
    """Cover both language prefixes, clone modes, reference lengths, and instructions."""
    for language in ("auto", "english"):
      for icl in (False, True):
        for frames in (4, 620):
          for instructions in (0, 12):
            with self.subTest(language=language, icl=icl, frames=frames, inst=instructions):
              prompt = VoiceClonePrompt(
                ref_code=torch.zeros((frames, 16), dtype=torch.long),
                ref_spk_embedding=torch.ones(8),
                ref_text="reference",
                icl_mode=icl,
                x_vector_only_mode=not icl,
              )
              prepared = EncodedText(
                request=SynthesisRequest(
                  text="target",
                  language=language,
                  voice_clone_prompt=prompt,
                ),
                text_token_ids=torch.arange(20),
                instruct_token_ids=torch.arange(instructions),
                ref_token_ids=torch.arange(15),
              )
              executor = _base_executor()
              actual = executor._prepare_base_input_plan(prepared).sequence_lengths[0]
              self.assertEqual(base_prefill_tokens(prepared), actual)

  def test_long_text_is_lossless_and_each_part_fits(self) -> None:
    """Variable-width UTF-8 text exercises budgets not equivalent to character counts."""
    for text in ("Hello world. " * 100, "你好世界" * 100, "a" * 1000):

      def measure(part: str) -> int:
        """Model a variable-width token budget with fixed reference overhead."""
        return 600 + len(part.encode("utf-8"))

      parts = split_to_budget(text, measure, 640)
      self.assertEqual("".join(parts), text)
      self.assertTrue(all(measure(part) <= 640 for part in parts))
      self.assertTrue(all(part.strip() for part in parts))

  def test_exact_capacity_is_not_split(self) -> None:
    """Leave requests that exactly fit unchanged."""
    self.assertEqual(split_to_budget("hello", len, 5), ["hello"])

  def test_oversized_reference_is_actionable(self) -> None:
    """Do not loop forever or drop text when fixed conditioning cannot fit."""
    with self.assertRaises(ValidationError) as raised:
      split_to_budget("hello world", lambda text: 641 + len(text), 640)
    self.assertEqual(raised.exception.status, 400)
    self.assertIn("shorter reference", raised.exception.detail)

  def test_request_rejection_maps_to_client_error(self) -> None:
    """Nari admission rejections must not leak through as generic HTTP 500 errors."""
    from nari_qwen3_tts.contract.errors import RequestRejected

    from app.backends.pytorch.nari_tts import NariTTSRuntime

    runtime = object.__new__(NariTTSRuntime)
    error = runtime._translate_error(RequestRejected("Unsupported capture schedule"))
    assert isinstance(error, ValidationError)
    self.assertEqual(error.status, 400)

  def test_whitespace_does_not_create_empty_synthesis(self) -> None:
    """Pathological whitespace returns a domain error, never an empty Nari request."""

    def measure(text: str) -> int:
      """Reject empty synthesis inputs as the real Nari contract does."""
      self.assertTrue(text.strip())
      return len(text)

    with self.assertRaises(ValidationError):
      split_to_budget("ab" + " " * 100 + "cd", measure, 10)


if __name__ == "__main__":
  unittest.main()
