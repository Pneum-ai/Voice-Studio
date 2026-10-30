"""Shared Qwen generation limits, language names, and cancellation controls."""

from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

LOGGER = logging.getLogger(__name__)
LANGUAGE_NAMES = {
  "auto": "Auto",
  "de": "German",
  "en": "English",
  "es": "Spanish",
  "fr": "French",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "pt": "Portuguese",
  "ru": "Russian",
  "zh": "Chinese",
}
QWEN_MIN_NEW_TOKENS = 96
QWEN_MAX_NEW_TOKENS = 512
QWEN_TOKEN_HEADROOM = 48
QWEN_TOKENS_PER_WORD = 8
QWEN_TOKENS_PER_CHARACTER = 1.25


def qwen_generation_token_limit(text: str) -> int:
  """Bound Qwen audio tokens using conservative speech-length estimates.

  Qwen emits audio codes at 12 Hz. The upstream 2,048-token default permits
  nearly three minutes of audio when the model misses EOS, which can take even
  longer than real time to decode. Normal Pipecat speech is sentence-chunked,
  so a 512-token absolute ceiling leaves ample room without allowing runaway
  generations.
  """
  clean_text = str(text or "").strip()
  word_count = len(re.findall(r"\w+", clean_text, flags=re.UNICODE))
  estimated = QWEN_TOKEN_HEADROOM + max(
    word_count * QWEN_TOKENS_PER_WORD,
    math.ceil(len(clean_text) * QWEN_TOKENS_PER_CHARACTER),
  )
  return max(QWEN_MIN_NEW_TOKENS, min(QWEN_MAX_NEW_TOKENS, estimated))


@contextmanager
def cancelable_qwen_generation(
  model: Any,
  cancel_signal: threading.Event,
) -> Iterator[None]:
  """Inject cooperative cancellation into the upstream talker token loop."""
  core_model = getattr(model, "model", None)
  talker = getattr(core_model, "talker", None)
  if talker is None:
    LOGGER.warning("Qwen talker generation could not be made cooperatively cancellable.")
    yield
    return
  original_generate = getattr(talker, "generate", None)
  if not callable(original_generate):
    LOGGER.warning("Qwen talker generation could not be made cooperatively cancellable.")
    yield
    return

  from transformers import StoppingCriteria, StoppingCriteriaList

  class CancellationStoppingCriteria(StoppingCriteria):
    """Stop every sequence in this batch once all callers have cancelled."""

    def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
      """Return one stop flag per sequence from the shared cancellation event."""
      _ = (scores, kwargs)
      return input_ids.new_full(
        (input_ids.shape[0],),
        cancel_signal.is_set(),
      ).bool()

  cancellation_criteria = CancellationStoppingCriteria()

  def controlled_generate(*args: Any, **kwargs: Any):
    """Forward generation after appending the cooperative stopping criterion."""
    existing = kwargs.get("stopping_criteria")
    criteria = StoppingCriteriaList(list(existing or ()))
    criteria.append(cancellation_criteria)
    kwargs["stopping_criteria"] = criteria
    return original_generate(*args, **kwargs)

  talker.generate = controlled_generate
  try:
    yield
  finally:
    if getattr(talker, "generate", None) is controlled_generate:
      talker.generate = original_generate


__all__ = [
  "LANGUAGE_NAMES",
  "cancelable_qwen_generation",
  "qwen_generation_token_limit",
]
