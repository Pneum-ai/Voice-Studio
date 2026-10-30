"""Build, safely persist, and invalidate combined profile voice prompts."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from safetensors import SafetensorError
from sqlalchemy.ext.asyncio import AsyncSession

from app.backends.factory import BackendBundle
from app.core.config import Settings
from app.core.errors import ValidationError
from app.domain.entities import StoredSample
from app.persistence.profiles import ProfileRepository
from app.utils.audio import AudioData, load_audio, normalize_audio, write_wav
from app.utils.files import promote_file, resolve_data_path, unlink_if_exists
from app.utils.prompt_cache import load_qwen_prompt, save_qwen_prompt, tag_qwen_prompt

LOGGER = logging.getLogger(__name__)


class PromptService:
  """Create one backend prompt from every sample attached to a profile."""

  def __init__(
    self,
    settings: Settings,
    repository: ProfileRepository,
    backends: BackendBundle,
  ) -> None:
    """Initialize an isolated in-memory cache over safe persisted artifacts."""
    self.settings = settings
    self.repository = repository
    self.backends = backends
    self._memory: dict[str, object] = {}
    self._profile_keys: dict[str, set[str]] = {}
    self._lock = asyncio.Lock()

  async def get_prompt(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    model_id: str,
  ) -> object:
    """Return a cached prompt or build it from the profile's current samples."""
    samples = await self.repository.list_samples(session, profile_id)
    # Prompt construction can take seconds; release SQLite before inference work.
    await session.rollback()
    if not samples:
      raise ValidationError(
        "The selected profile has no reference samples.",
        "profile_has_no_samples",
      )
    digest = self._cache_digest(profile_id, model_id, samples)
    cache_key = f"{profile_id}-{digest}"
    async with self._lock:
      cached = self._memory.get(cache_key)
      if cached is not None:
        return cached
      self._profile_keys.setdefault(profile_id, set()).add(cache_key)
      prompt_path = self.settings.prompt_cache_dir / cache_key
      if self.backends.name == "pytorch":
        restored = await self._restore_qwen_prompt(prompt_path)
        if restored is not None:
          restored = tag_qwen_prompt(restored, cache_key=cache_key)
          self._memory[cache_key] = restored
          return restored

      reference_path, transcript = await asyncio.to_thread(
        self._build_reference,
        profile_id,
        digest,
        samples,
      )
      prompt = await self.backends.tts.create_prompt(reference_path, transcript)
      if self.backends.name == "pytorch":
        prompt = tag_qwen_prompt(prompt, cache_key=cache_key)
      self._memory[cache_key] = prompt
      if self.backends.name == "pytorch":
        await asyncio.to_thread(save_qwen_prompt, prompt_path, prompt)
      return prompt

  @staticmethod
  def _cache_digest(
    profile_id: str,
    model_id: str,
    samples: Sequence[StoredSample],
  ) -> str:
    """Hash all prompt-affecting metadata into a stable cache identifier."""
    digest = hashlib.sha256()
    digest.update(profile_id.encode())
    digest.update(model_id.encode())
    for sample in samples:
      digest.update(sample.id.encode())
      digest.update(sample.content_sha256.encode())
      digest.update(sample.transcript.encode())
    return digest.hexdigest()[:32]

  def _build_reference(
    self,
    profile_id: str,
    digest: str,
    samples: Sequence[StoredSample],
  ) -> tuple[Path, str]:
    """Return a single sample or atomically build a concatenated reference WAV."""
    transcripts = [sample.transcript for sample in samples]
    if len(samples) == 1:
      audio_file = samples[0].audio_file
      return resolve_data_path(self.settings.data_dir, audio_file), transcripts[0]

    destination = self.settings.reference_cache_dir / f"{profile_id}-{digest}.wav"
    if not destination.is_file():
      audio_parts = [
        normalize_audio(
          load_audio(
            resolve_data_path(
              self.settings.data_dir,
              sample.audio_file,
            )
          )
        ).samples
        for sample in samples
      ]
      combined = normalize_audio(AudioData(np.concatenate(audio_parts).astype(np.float32), 24_000))
      descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{profile_id}-",
        suffix=".wav",
        dir=destination.parent,
      )
      os.close(descriptor)
      staged = Path(staged_name)
      try:
        write_wav(staged, combined)
        promote_file(staged, destination)
      finally:
        unlink_if_exists(staged)
    return destination, " ".join(transcripts)

  async def _restore_qwen_prompt(self, path: Path) -> object | None:
    """Restore a safe PyTorch prompt, discarding incomplete cache pairs."""
    json_path = path.with_suffix(".json")
    tensor_path = path.with_suffix(".safetensors")
    if not json_path.is_file() or not tensor_path.is_file():
      await asyncio.gather(
        asyncio.to_thread(unlink_if_exists, json_path),
        asyncio.to_thread(unlink_if_exists, tensor_path),
      )
      return None
    try:
      return await asyncio.to_thread(load_qwen_prompt, path)
    except (ImportError, OSError, SafetensorError, TypeError, ValueError) as error:
      LOGGER.warning("Discarding invalid prompt cache %s: %s", path.name, error)
      await asyncio.gather(
        asyncio.to_thread(unlink_if_exists, json_path),
        asyncio.to_thread(unlink_if_exists, tensor_path),
      )
      return None

  async def clear_profile(self, profile_id: str) -> int:
    """Delete in-memory and persisted prompt artifacts for one profile."""
    async with self._lock:
      for key in self._profile_keys.pop(profile_id, set()):
        self._memory.pop(key, None)
      return await asyncio.to_thread(self._delete_matching, f"{profile_id}-*")

  async def clear_all(self) -> int:
    """Delete every derived prompt and reference artifact."""
    async with self._lock:
      self._memory.clear()
      self._profile_keys.clear()
      return await asyncio.to_thread(self._delete_matching, "*")

  def _delete_matching(self, pattern: str) -> int:
    """Delete regular cache files matching a constrained filename pattern."""
    deleted = 0
    for directory in (
      self.settings.prompt_cache_dir,
      self.settings.reference_cache_dir,
    ):
      for path in directory.glob(pattern):
        if path.is_file() and unlink_if_exists(path):
          deleted += 1
    return deleted
