"""Safe shorthand parsing and stable voice-style conditioning."""

from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.services.style_controls import prepare_style_input, style_instruction


def test_leading_styles_are_conditioned_and_removed_from_spoken_text() -> None:
  """Keep style tags out of speech while retaining explicit pause positions."""
  text, controls = prepare_style_input(
    "<happy> <slow> We made it <pause> together.",
    (),
  )
  assert text == "We made it <pause> together."
  assert controls == ("emotion_happy", "pace_slow", "pause_explicit")
  instruction = style_instruction(None, controls)
  assert instruction is not None
  assert "<happy>" in instruction
  assert "<slow>" in instruction
  assert "never speak the tag" in instruction


def test_conflicting_style_dimension_is_rejected() -> None:
  """Avoid ambiguous examples with two target emotions."""
  with pytest.raises(ValidationError, match="one emotion"):
    prepare_style_input("Hello", ("emotion_happy", "emotion_sad"))
