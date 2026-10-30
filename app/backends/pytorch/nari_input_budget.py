"""Budget Base ICL prompts without allocating GPU embeddings or truncating speech."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.core.errors import ValidationError

if TYPE_CHECKING:
  from nari_qwen3_tts.contract.request import EncodedText


def base_prefill_tokens(prepared: EncodedText) -> int:
  """Count the vendored Base full-text layout, including reference and instruction tokens.

  Keep this in parity with TalkerExecutor._prepare_base_input_plan. The role and
  codec prefix occupy eight positions for auto language, nine for explicit language.
  Target/reference wrappers lose eight/five tokens; EOS and codec BOS add two.
  """
  prompt = prepared.request.voice_clone_prompt
  if prompt is None or not prepared.request.non_streaming_mode:
    raise ValueError("Token budgeting requires a full-text Base clone request.")
  total = 8 if prepared.request.language == "auto" else 9
  total += prepared.text_token_ids[3:-5].numel() + 2
  total += prepared.instruct_token_ids.numel()
  if prompt.icl_mode:
    if prepared.ref_token_ids is None or prompt.ref_code is None:
      raise ValueError("ICL token budgeting requires reference text and audio codes.")
    total += prepared.ref_token_ids[3:-2].numel() + prompt.ref_code.shape[0]
  return int(total)


def split_to_budget(text: str, measure: Callable[[str], int], capacity: int) -> list[str]:
  """Preflight all text segments, preferring word boundaries without losing characters.

  Measure every candidate with the runtime tokenizer rather than estimating from
  character count. Raise a domain error before any audio if even one character
  cannot fit alongside the unchanged voice reference and instructions.
  """
  pending = [text]
  result: list[str] = []
  while pending:
    part = pending.pop()
    if measure(part) <= capacity:
      result.append(part)
      continue
    if len(part.strip()) <= 1:
      raise ValidationError(
        "The voice reference and instructions leave insufficient speech token capacity. "
        "Use a shorter reference sample or shorter instructions.",
      )
    # A long trailing whitespace run must not produce a whitespace-only request.
    start = len(part) - len(part.lstrip())
    end = len(part.rstrip())
    middle = max(start + 1, min(len(part) // 2, end - 1))
    boundaries = [
      match.end()
      for match in re.finditer(r"\s+", part)
      if part[: match.end()].strip() and part[match.end() :].strip()
    ]
    cut = min(boundaries, key=lambda index: abs(index - middle)) if boundaries else middle
    pending.extend((part[cut:], part[:cut]))
  return result
