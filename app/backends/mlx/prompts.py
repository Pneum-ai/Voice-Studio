"""Per-row preset and reference conditioning for mixed-voice MLX batches."""

from __future__ import annotations

from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from app.backends.pytorch.qwen_generation import LANGUAGE_NAMES


class ReferenceModel:
  """Route upstream prompt math through a session-local speaker embedding cache."""

  def __init__(
    self, model: Any, reference: str, transcript: str, *, cache_speaker: bool = False
  ) -> None:
    """Normalize one reference on the GPU owner; cache only this exact waveform."""
    self.model = model
    self.reference, self.transcript = model._normalize_shared_batch_refs(
      1, ref_audio=reference, ref_text=transcript
    )
    self.cache_speaker = cache_speaker
    self.speaker_embedding: Any = None

  def __getattr__(self, name: str) -> Any:
    """Delegate weights and model configuration without mutating the shared model."""
    return getattr(self.model, name)

  def _prepare_batch_inputs(self, *args: Any, **kwargs: Any) -> Any:
    """Prepare reference-conditioned inputs using the original model implementation."""
    return self.model._prepare_batch_inputs.__func__(
      self, *args, **kwargs, ref_audio=self.reference, ref_text=self.transcript
    )

  def _prepare_icl_generation_inputs(self, *args: Any, **kwargs: Any) -> Any:
    """Route speaker extraction through this reference's cache."""
    return self.model._prepare_icl_generation_inputs.__func__(self, *args, **kwargs)

  def extract_speaker_embedding(self, audio: Any) -> Any:
    """Never reuse an embedding for another reference object."""
    if not self.cache_speaker or audio is not self.reference:
      return self.model.extract_speaker_embedding(audio)
    if self.speaker_embedding is None:
      self.speaker_embedding = self.model.extract_speaker_embedding(audio)
    return self.speaker_embedding


class PromptModel:
  """Build padded batches with independent voice and language conditioning per row."""

  def __init__(self, model: Any, *, cache_speaker: bool) -> None:
    """Retain at most sixteen normalized references for the current serving session."""
    self.model = model
    self.cache_speaker = cache_speaker
    self.references: OrderedDict[tuple[str, str, int, int], ReferenceModel] = OrderedDict()
    self.preparing: deque[Any] = deque()

  def __getattr__(self, name: str) -> Any:
    """Share model math and weights with the upstream continuous session."""
    return getattr(self.model, name)

  def _prepare_batch_inputs(self, *args: Any, **kwargs: Any) -> Any:
    """Use upstream padding with the selected model’s reference or preset input layout."""
    try:
      if self.model.config.tts_model_type == "base":
        kwargs.update(ref_audio=True, ref_text="per-request")
      return self.model._prepare_batch_inputs.__func__(self, *args, **kwargs)
    finally:
      self.preparing.clear()
      # Upstream caches reference codes by fingerprint; bound that cache as well.
      cache = self.model._icl_cache
      while len(cache) > 16:
        del cache[next(iter(cache))]

  def _prepare_generation_inputs(self, text: str, **kwargs: Any) -> Any:
    """Use each preset row's language while upstream supplies its speaker and instructions."""
    item = self.preparing.popleft()
    language = item.extra["language"]
    kwargs["language"] = LANGUAGE_NAMES.get(language.casefold(), language)
    return self.model._prepare_generation_inputs(text, **kwargs)

  def _prepare_icl_generation_inputs(self, text: str, **kwargs: Any) -> Any:
    """Resolve exactly the row selected by upstream; text is never used as a cache key."""
    del kwargs
    item = self.preparing.popleft()
    path = Path(item.ref_audio)
    stat = path.stat()
    key = (str(path), item.ref_text, stat.st_mtime_ns, stat.st_size)
    reference = self.references.get(key)
    if reference is None:
      reference = ReferenceModel(
        self.model, str(path), item.ref_text, cache_speaker=self.cache_speaker
      )
      self.references[key] = reference
    self.references.move_to_end(key)
    while len(self.references) > 16:
      self.references.popitem(last=False)
    return reference._prepare_icl_generation_inputs(
      text,
      ref_audio=reference.reference,
      ref_text=reference.transcript,
      language=LANGUAGE_NAMES.get(item.extra["language"].casefold(), item.extra["language"]),
    )

  def clear(self) -> None:
    """Release retained reference audio and speaker embeddings on the GPU owner."""
    self.preparing.clear()
    self.references.clear()
