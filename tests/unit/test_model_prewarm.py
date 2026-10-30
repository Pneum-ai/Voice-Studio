"""Default model prewarming behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from app.core.config import Settings
from app.services.jobs import JobService
from app.services.models import ModelService
from tests.helpers import fake_backends


@pytest.mark.asyncio
async def test_prewarm_downloads_and_loads_both_defaults(tmp_path: Path) -> None:
  """Load both defaults at boot even when their weights are not cached yet."""
  backends = fake_backends()
  tts = cast(Any, backends.tts)
  stt = cast(Any, backends.stt)
  tts.cached.clear()
  stt.cached.clear()
  tts._loaded_model = None
  stt._loaded_model = None
  service = ModelService(
    Settings(
      prewarm=True,
      default_qwen_model="qwen-tts-1.7B",
      model_selection_file=tmp_path / "model-selection.json",
    ),
    backends,
    JobService(300),
  )

  await service.prewarm_defaults()

  assert tts.loaded_model == "qwen-tts-1.7B"
  assert stt.loaded_model == "whisper-turbo"
  assert "qwen-tts-1.7B" in tts.cached
  assert "whisper-turbo" in stt.cached


@pytest.mark.asyncio
async def test_no_prewarm_leaves_both_defaults_unloaded() -> None:
  """Do no model downloads or loads when startup prewarming is disabled."""
  backends = fake_backends()
  tts = cast(Any, backends.tts)
  stt = cast(Any, backends.stt)
  tts.cached.clear()
  stt.cached.clear()
  tts._loaded_model = None
  stt._loaded_model = None
  service = ModelService(Settings(prewarm=False), backends, JobService(300))

  await service.prewarm_defaults()

  assert tts.loaded_model is None
  assert stt.loaded_model is None
  assert not tts.cached
  assert not stt.cached
