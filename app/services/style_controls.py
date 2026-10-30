"""Structured voice-style controls shared by speech and profile samples."""

from __future__ import annotations

import re

from app.core.errors import ValidationError

CONTROL_GROUPS = {
  "emotion_happy": "emotion",
  "emotion_sad": "emotion",
  "emotion_calm": "emotion",
  "emotion_excited": "emotion",
  "emotion_serious": "emotion",
  "pace_slow": "pace",
  "pace_fast": "pace",
  "energy_low": "energy",
  "energy_high": "energy",
  "pauses_more": "pauses",
  "pauses_fewer": "pauses",
  "pauses_long": "pauses",
  "pauses_short": "pauses",
  "pause_explicit": "inline_pause",
}
CONTROL_TAGS = {
  "happy": "emotion_happy",
  "sad": "emotion_sad",
  "calm": "emotion_calm",
  "excited": "emotion_excited",
  "serious": "emotion_serious",
  "slow": "pace_slow",
  "fast": "pace_fast",
  "soft": "energy_low",
  "strong": "energy_high",
  "more-pauses": "pauses_more",
  "fewer-pauses": "pauses_fewer",
  "long-pauses": "pauses_long",
  "short-pauses": "pauses_short",
}
CONTROL_DESCRIPTIONS = {
  "emotion_happy": "Use a genuinely happy emotional tone.",
  "emotion_sad": "Use a sad, subdued emotional tone.",
  "emotion_calm": "Sound calm and composed.",
  "emotion_excited": "Sound excited and enthusiastic.",
  "emotion_serious": "Use a serious, grounded tone.",
  "pace_slow": "Speak at a slower pace.",
  "pace_fast": "Speak at a faster pace.",
  "energy_low": "Use softer, lower-energy delivery.",
  "energy_high": "Use stronger, higher-energy delivery.",
  "pauses_more": "Use more natural pauses.",
  "pauses_fewer": "Use fewer pauses.",
  "pauses_long": "Make pauses longer.",
  "pauses_short": "Keep pauses brief.",
  "pause_explicit": "Treat every <pause> marker as silence and never speak the tag.",
}
CONTROL_CANONICAL_TAGS = {value: key for key, value in CONTROL_TAGS.items()}
CONTROL_CANONICAL_TAGS["pause_explicit"] = "pause"
LEADING_TAG = re.compile(r"^\s*<([a-z-]+)>\s*", re.IGNORECASE)
INLINE_PAUSE = re.compile(r"<pause>", re.IGNORECASE)


def clean_style_controls(controls: tuple[str, ...] | list[str]) -> tuple[str, ...]:
  """Validate one optional control per style dimension."""
  cleaned = tuple(dict.fromkeys(item.strip().casefold() for item in controls if item.strip()))
  invalid = [item for item in cleaned if item not in CONTROL_GROUPS]
  if invalid:
    raise ValidationError(f"Unknown voice style control: {invalid[0]}")
  groups: set[str] = set()
  for item in cleaned:
    group = CONTROL_GROUPS[item]
    if group in groups:
      raise ValidationError(f"Choose only one {group} voice style.")
    groups.add(group)
  return cleaned


def parse_style_tags(text: str) -> tuple[str, tuple[str, ...]]:
  """Remove supported leading tags so they are conditioned, never spoken."""
  remaining = text
  controls: list[str] = []
  while match := LEADING_TAG.match(remaining):
    tag = match.group(1).casefold()
    control = CONTROL_TAGS.get(tag)
    if control is None:
      break
    controls.append(control)
    remaining = remaining[match.end() :]
  return remaining.strip(), clean_style_controls(controls)


def merge_style_controls(
  controls: tuple[str, ...] | list[str],
  overrides: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
  """Apply newer controls by dimension while retaining unrelated choices."""
  merged = {CONTROL_GROUPS[item]: item for item in clean_style_controls(controls)}
  for item in clean_style_controls(overrides):
    merged[CONTROL_GROUPS[item]] = item
  order = ("emotion", "pace", "energy", "pauses", "inline_pause")
  return tuple(merged[group] for group in order if group in merged)


def prepare_style_input(
  text: str,
  controls: tuple[str, ...] | list[str] = (),
) -> tuple[str, tuple[str, ...]]:
  """Merge explicit controls with safe leading shorthand tags."""
  spoken_text, tagged = parse_style_tags(text)
  if not spoken_text:
    raise ValidationError("Speech text must contain words after voice style tags.")
  inline = ("pause_explicit",) if INLINE_PAUSE.search(spoken_text) else ()
  return spoken_text, merge_style_controls(merge_style_controls(controls, tagged), inline)


def style_instruction(
  instructions: str | None,
  controls: tuple[str, ...] | list[str],
) -> str | None:
  """Build the stable Qwen instruction used for inference."""
  cleaned = clean_style_controls(controls)
  if not cleaned:
    return instructions
  tags = " ".join(f"<{CONTROL_CANONICAL_TAGS[item]}>" for item in cleaned)
  descriptions = " ".join(CONTROL_DESCRIPTIONS[item] for item in cleaned)
  control_prompt = f"Voice style controls: {tags}. {descriptions}"
  return f"{control_prompt} {instructions}" if instructions else control_prompt
