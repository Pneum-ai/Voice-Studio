"""Nari CUDA-graph serving for Qwen3-TTS Base and CustomVoice."""

from __future__ import annotations

import asyncio
import logging
import platform
import threading
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.protocols import BatchGenerationRequest
from app.backends.pytorch.nari_diagnostics import record_nari_diagnostics
from app.backends.pytorch.nari_environment import configure_cuda_toolkit
from app.backends.pytorch.nari_input_budget import base_prefill_tokens, split_to_budget
from app.backends.pytorch.nari_policy import build_nari_engine_with_policy
from app.backends.pytorch.nari_streaming import (
  AdaptiveAdmission,
  AsyncPCMBridge,
  LeadingSilenceTrimmer,
  NariRuntimeConfig,
  NariRuntimeStatus,
  NariWatchdog,
  iter_with_deadline,
)
from app.backends.pytorch.qwen_generation import LANGUAGE_NAMES, qwen_generation_token_limit
from app.core.errors import (
  InferenceTimeoutError,
  InferenceUnavailableError,
  OverloadedError,
  ServiceError,
  ValidationError,
)
from app.domain.speech import BuiltinVoicePrompt
from app.utils.audio import (
  TARGET_SAMPLE_RATE,
  AudioData,
  PCM16AudioData,
  StreamAudioData,
  load_audio,
)

LOGGER = logging.getLogger(__name__)


class NariTTSRuntime:
  """Own the upstream Nari scheduler, captured executors, and prompt assets."""

  def __init__(
    self,
    repository: Path,
    *,
    device: str,
    model_id: str | None = None,
    config: NariRuntimeConfig | None = None,
  ) -> None:
    """Load Qwen weights and capture Nari execution."""
    configure_cuda_toolkit()
    from nari_qwen3_tts.config import ModelAssetConfig
    from nari_qwen3_tts.model.public import Qwen3TTSModel

    self._torch = __import__("torch")
    self._config = config or NariRuntimeConfig()
    resolved_device = "cuda:0" if device == "cuda" else device
    self._device = self._torch.device(resolved_device)
    self._model = Qwen3TTSModel(
      ModelAssetConfig(
        model_id=str(repository),
        device=resolved_device,
        require_h100=False,
        local_files_only=True,
        blackwell_fp8=self._config.blackwell_fp8,
      )
    )
    self._assets = self._model.load_assets()
    policy = build_nari_engine_with_policy(
      self._model,
      self._assets,
      self._config,
      model_id=model_id,
      torch_module=self._torch,
      device=self._device,
    )
    self._engine = policy.engine
    self._config = policy.runtime_config
    self._policy = policy
    self._admission = AdaptiveAdmission(self._config, model_id=model_id)
    self._watchdog: NariWatchdog | None = None
    self._start_lock = threading.Lock()
    self._started = False
    self._startup_failure_id: str | None = None
    self._startup_failure_detail: str | None = None

  def runtime_status(self) -> NariRuntimeStatus:
    """Return live engine readiness and model-aware admission pressure."""
    if self._watchdog is not None:
      status = self._watchdog.status()
    else:
      active, queued, limit, ceiling = self._admission.snapshot
      failed = self._startup_failure_id is not None
      status = NariRuntimeStatus(
        ready=not self._started and not failed,
        reason="startup_failed" if failed else "ready_unstarted",
        active=active,
        queued=queued,
        active_limit=limit,
        active_ceiling=ceiling,
        failure_id=self._startup_failure_id,
        failure_detail=self._startup_failure_detail,
      )
    return replace(
      status,
      policy_name=self._policy.policy_name,
      capture_profile=self._policy.capture_profile,
      capture_profile_sha256=self._policy.capture_profile_sha256,
      capture_fallbacks=self._policy.capture_fallbacks,
    )

  @staticmethod
  def supported(*, device: str) -> tuple[bool, str]:
    """Return whether Nari's published CUDA kernel stack can run on this host."""
    if platform.system() != "Linux":
      return False, "Nari FlashInfer/Triton execution currently requires Linux or WSL2"
    if not device.startswith("cuda"):
      return False, "Nari execution requires CUDA"
    try:
      __import__("nari_qwen3_tts")
      __import__("flashinfer")
      __import__("triton")
    except ImportError as error:
      return False, f"Nari CUDA dependencies are unavailable: {error}"
    return True, "available"

  def create_prompt(self, audio_path: Path, transcript: str) -> object:
    """Encode reference speech once for Base ICL and speaker conditioning."""
    from app.backends.pytorch.qwen_runtime import VoiceClonePromptItem, mel_spectrogram

    audio = load_audio(audio_path, sample_rate=TARGET_SAMPLE_RATE)
    tokenizer: Any = self._assets.speech_tokenizer
    encoded = tokenizer.encode([audio.samples], sr=audio.sample_rate)
    ref_code = encoded.audio_codes[0].to(device=self._device, dtype=self._torch.long)
    encoder = self._assets.speaker_encoder
    if encoder is None:
      raise RuntimeError("The selected Nari checkpoint does not contain a speaker encoder.")
    waveform = self._torch.from_numpy(audio.samples).unsqueeze(0)
    mel = mel_spectrogram(
      waveform,
      n_fft=1024,
      num_mels=128,
      sampling_rate=TARGET_SAMPLE_RATE,
      hop_size=256,
      win_size=1024,
      fmin=0,
      fmax=12000,
    ).transpose(1, 2)
    dtype = next(encoder.parameters()).dtype
    embedding = encoder(mel.to(device=self._device, dtype=dtype))[0].detach()
    return [
      VoiceClonePromptItem(
        ref_code=ref_code,
        ref_spk_embedding=embedding,
        ref_text=transcript,
        x_vector_only_mode=False,
        icl_mode=True,
      )
    ]

  def _request(
    self,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    low_latency: bool,
  ) -> Any:
    """Translate cloned or built-in conditioning into Nari’s synthesis contract."""
    from nari_qwen3_tts.contract.request import (
      SynthesisRequest,
      VoiceClonePrompt,
    )

    if isinstance(request.prompt, BuiltinVoicePrompt):
      return SynthesisRequest(
        text=request.text,
        voice=request.prompt.speaker,
        language=LANGUAGE_NAMES.get(request.language.casefold(), request.language).lower(),
        instruct=request.instructions or "",
        non_streaming_mode=True,
        random_seed=0 if seed is None else seed,
        max_new_tokens=qwen_generation_token_limit(request.text),
        skip_fixed_bootstrap_audio=True,
        stream_chunk_schedule=self._config.chunk_schedule_for(low_latency=low_latency),
      )
    prompt_items = request.prompt if isinstance(request.prompt, list) else [request.prompt]
    if len(prompt_items) != 1:
      raise TypeError("Nari generation requires exactly one voice-clone prompt item.")
    prompt = prompt_items[0]
    ref_code = getattr(prompt, "ref_code", None)
    embedding = getattr(prompt, "ref_spk_embedding", None)
    ref_text = getattr(prompt, "ref_text", None)
    if ref_code is None or embedding is None or not ref_text:
      raise TypeError("Nari Base ICL requires reference codes, embedding, and text.")
    language = LANGUAGE_NAMES.get(request.language.casefold(), request.language).lower()
    return SynthesisRequest(
      text=request.text,
      voice="aiden",
      language=language,
      instruct=request.instructions or "",
      non_streaming_mode=True,
      random_seed=0 if seed is None else seed,
      max_new_tokens=qwen_generation_token_limit(request.text),
      skip_fixed_bootstrap_audio=False,
      stream_chunk_schedule=self._config.chunk_schedule_for(low_latency=low_latency),
      voice_clone_prompt=VoiceClonePrompt(
        ref_code=ref_code.to(device=self._device, dtype=self._torch.long),
        ref_spk_embedding=embedding.to(device=self._device),
        ref_text=ref_text,
        cache_key=getattr(request.prompt, "cache_key", None),
        x_vector_only_mode=False,
        icl_mode=True,
      ),
    )

  def _ensure_started(self, request: Any) -> None:
    """Run Nari's captured ordinary-request readiness proof exactly once."""
    if self._started:
      return
    with self._start_lock:
      if self._started:
        return
      if self._startup_failure_id is not None:
        raise RuntimeError(
          f"Nari startup previously failed (failure_id={self._startup_failure_id})."
        )
      prompt = request.voice_clone_prompt
      readiness_prompt = (
        replace(
          prompt,
          ref_code=None,
          ref_text=None,
          x_vector_only_mode=True,
          icl_mode=False,
        )
        if prompt is not None
        else None
      )
      readiness = replace(
        request,
        text="The speech engine is ready.",
        do_sample=False,
        subtalker_dosample=False,
        max_new_tokens=4,
        ignore_eos=True,
        skip_fixed_bootstrap_audio=False,
        stream_chunk_schedule=(1,),
        voice_clone_prompt=readiness_prompt,
      )
      try:
        self._engine.start(readiness, timeout_s=1800.0)
      except BaseException as error:
        self._startup_failure_id = uuid.uuid4().hex[:12]
        self._startup_failure_detail = f"{type(error).__name__}: {error}"
        LOGGER.exception(
          "Nari startup failed (failure_id=%s)",
          self._startup_failure_id,
          exc_info=error,
        )
        raise
      self._started = True
      self._watchdog = NariWatchdog(self._engine, self._admission, self._config)
      self._watchdog.start()

  def open_stream(
    self,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    low_latency: bool,
  ) -> tuple[str, Any]:
    """Submit one synthesis request and return its blocking PCM stream."""
    synthesis = self._request(request, seed=seed, low_latency=low_latency)
    self._ensure_started(synthesis)
    request_id = f"voice-open-{uuid.uuid4().hex}"
    return request_id, self._engine.submit(request_id, synthesis)

  async def open_live_text(
    self,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> Any:
    """Create one append-safe Base request without submitting empty text."""
    from app.backends.pytorch.nari_live import NariLiveTextSession

    session = NariLiveTextSession(
      self,
      request,
      seed=seed,
      cancel_event=cancel_event,
    )
    await session.prepare()
    return session

  def cancel(self, request_id: str) -> None:
    """Cancel one active request without masking its original stream failure."""
    try:
      self._engine.cancel(request_id)
    except KeyError:
      return
    except Exception as error:
      LOGGER.warning("Best-effort Nari cancellation failed for %s: %s", request_id, error)

  def _translate_error(self, error: Exception) -> Exception:
    """Map Nari capacity and terminal failures onto stable Voice Open errors."""
    from nari_qwen3_tts.contract.errors import (
      RequestRejected,
      ServiceCapacityExceeded,
      ServiceUnavailable,
    )

    if isinstance(error, RequestRejected):
      return ValidationError("The speech input is not supported by the active Nari profile.")
    if isinstance(error, ServiceCapacityExceeded):
      return OverloadedError(self._config.queue_timeout_ms)
    if isinstance(error, ServiceUnavailable):
      status = self.runtime_status()
      return InferenceUnavailableError(status.reason, status.failure_id)
    if isinstance(error, TimeoutError):
      return InferenceTimeoutError("Nari command submission exceeded its deadline.")
    return error

  async def stream(
    self,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Preflight token-sized segments and stream them sequentially without losing text."""
    if cancel_event.is_set():
      return
    parts = await asyncio.to_thread(self._split_request, request, seed=seed)
    for text in parts:
      if cancel_event.is_set():
        return
      async for chunk in self._stream_one(
        replace(request, text=text),
        seed=seed,
        cancel_event=cancel_event,
      ):
        yield chunk

  def _split_request(self, request: BatchGenerationRequest, *, seed: int | None) -> list[str]:
    """Use the selected capture catalog and CPU tokenizer before scheduler admission."""
    synthesis = self._request(request, seed=seed, low_latency=False)
    capacity = max(key.token_capacity for key in self._engine.catalog.talker_prefill)

    def measure(text: str) -> int:
      """Count a full prompt without touching executor-owned caches."""
      prepared = self._model.prepare(replace(synthesis, text=text))
      if synthesis.voice_clone_prompt is not None:
        return base_prefill_tokens(prepared)
      from nari_qwen3_tts.model.input_layout import custom_voice_talker_input_seq_len

      return custom_voice_talker_input_seq_len(
        self._model.model_config,
        text_token_count=prepared.text_token_ids.numel(),
        instruct_token_count=prepared.instruct_token_ids.numel(),
        language=synthesis.language,
        speaker=synthesis.voice,
        non_streaming_mode=synthesis.non_streaming_mode,
      )

    return split_to_budget(request.text, measure, capacity)

  async def _stream_one(
    self,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Yield transport-ready chunks as Nari's stateful Codec publishes PCM."""
    lease = await self._admission.enter()
    request_id: str | None = None
    bridge: AsyncPCMBridge | None = None
    completed = False
    penalize = True
    first_audio_ms: float | None = None
    buffered_pcm_ms = 0
    started = asyncio.get_running_loop().time()
    trimmer = (
      LeadingSilenceTrimmer(
        sample_rate=TARGET_SAMPLE_RATE,
        threshold_dbfs=self._config.silence_threshold_dbfs,
        max_ms=self._config.silence_max_ms,
        preroll_ms=self._config.silence_preroll_ms,
        min_output_ms=self._config.silence_min_output_ms,
      )
      if self._config.silence_trim
      else None
    )
    try:
      request_id, pcm_stream = await asyncio.to_thread(
        self.open_stream,
        request,
        seed=seed,
        low_latency=lease.startup_reserved,
      )
      bridge = AsyncPCMBridge(pcm_stream)
      async for value in iter_with_deadline(
        bridge,
        first_audio_timeout_s=self._config.first_audio_timeout_seconds,
        request_timeout_s=self._config.request_timeout_seconds,
      ):
        if cancel_event.is_set():
          penalize = False
          break
        if first_audio_ms is None:
          first_audio_ms = (asyncio.get_running_loop().time() - started) * 1000
        if not value:
          continue
        buffered_pcm_ms += len(value) * 1000 // (2 * TARGET_SAMPLE_RATE)
        if buffered_pcm_ms >= self._config.silence_min_output_ms:
          await self._admission.release_startup(lease)
        if trimmer is None or trimmer.opened:
          yield PCM16AudioData(value, TARGET_SAMPLE_RATE)
          continue
        samples = np.frombuffer(value, dtype="<i2").astype(np.float32) / 32767.0
        output = trimmer.process(AudioData(samples, TARGET_SAMPLE_RATE))
        if output is not None and output.samples.size:
          yield output
      else:
        completed = True
      if completed and trimmer is not None:
        tail = trimmer.finish()
        if tail is not None and tail.samples.size:
          yield tail
      if completed and bridge is not None:
        record_nari_diagnostics(bridge.timing)
    except asyncio.CancelledError:
      penalize = False
      raise
    except ServiceError:
      raise
    except Exception as error:
      raise self._translate_error(error) from error
    finally:
      if bridge is not None:
        bridge.close()
      try:
        if request_id is not None and (cancel_event.is_set() or not completed):
          await asyncio.shield(asyncio.to_thread(self.cancel, request_id))
      finally:
        try:
          await self._admission.observe(first_audio_ms, failed=penalize and not completed)
        finally:
          await self._admission.leave(lease)

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None,
  ) -> list[AudioData]:
    """Collect concurrent Nari streams while its unified scheduler batches stages."""
    events = list(cancel_events or ())
    events.extend([None] * (len(requests) - len(events)))

    async def collect(index: int, request: BatchGenerationRequest) -> AudioData:
      event = events[index] or asyncio.Event()
      chunks = [chunk async for chunk in self.stream(request, seed=seed, cancel_event=event)]
      samples = (
        np.concatenate([chunk.samples for chunk in chunks])
        if chunks
        else np.empty(0, dtype=np.float32)
      )
      return AudioData(samples, TARGET_SAMPLE_RATE)

    tasks = (collect(index, item) for index, item in enumerate(requests))
    return list(await asyncio.gather(*tasks))

  def stop(self) -> None:
    """Stop the engine thread and release request-owned resources."""
    if self._watchdog is not None:
      self._watchdog.stop()
      self._watchdog = None
    if self._started:
      self._engine.stop(timeout_s=30.0)
      self._started = False


__all__ = ["NariTTSRuntime"]
