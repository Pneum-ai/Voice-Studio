"""Safely serialize Qwen voice-clone prompts without executable pickle data."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from app.utils.files import unlink_if_exists


class QwenPrompt(list[Any]):
  """Carry a stable profile digest alongside official Qwen prompt items."""

  def __init__(self, values: list[Any], *, cache_key: str) -> None:
    """Retain list compatibility while exposing an inference cache identity."""
    super().__init__(values)
    self.cache_key = cache_key


def tag_qwen_prompt(prompt: object, *, cache_key: str) -> object:
  """Attach a stable cache identity to an official list-shaped Qwen prompt."""
  if not isinstance(prompt, list):
    raise TypeError("Qwen prompt must be a list before cache tagging.")
  return QwenPrompt(prompt, cache_key=cache_key)


def save_qwen_prompt(path: Path, prompt: object) -> None:
  """Persist a Qwen prompt as validated JSON metadata and safe tensor data."""
  if not isinstance(prompt, list) or not prompt:
    raise TypeError("Qwen prompt must be a non-empty list.")
  path.parent.mkdir(parents=True, exist_ok=True)
  tensors: dict[str, Any] = {}
  items: list[dict[str, object]] = []
  for index, item in enumerate(prompt):
    embedding = getattr(item, "ref_spk_embedding", None)
    if embedding is None:
      raise TypeError("Qwen prompt item is missing its speaker embedding.")
    tensors[f"item_{index}_embedding"] = embedding.detach().cpu().contiguous()
    reference_code = getattr(item, "ref_code", None)
    if reference_code is not None:
      tensors[f"item_{index}_code"] = reference_code.detach().cpu().contiguous()
    items.append(
      {
        "has_ref_code": reference_code is not None,
        "icl_mode": bool(getattr(item, "icl_mode")),
        "ref_text": getattr(item, "ref_text", None),
        "x_vector_only_mode": bool(getattr(item, "x_vector_only_mode")),
      }
    )

  metadata = {"items": items, "schema_version": 1}
  tensor_path = path.with_suffix(".safetensors")
  json_path = path.with_suffix(".json")
  tensor_descriptor, tensor_name = tempfile.mkstemp(
    prefix=f".{path.name}-",
    suffix=".safetensors",
    dir=path.parent,
  )
  json_descriptor, json_name = tempfile.mkstemp(
    prefix=f".{path.name}-",
    suffix=".json",
    dir=path.parent,
  )
  os.close(tensor_descriptor)
  os.close(json_descriptor)
  staged_tensor = Path(tensor_name)
  staged_json = Path(json_name)
  try:
    save_file(tensors, str(staged_tensor))
    staged_json.write_text(
      json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
      encoding="utf-8",
    )
    os.replace(staged_tensor, tensor_path)
    os.replace(staged_json, json_path)
  finally:
    unlink_if_exists(staged_tensor)
    unlink_if_exists(staged_json)


def load_qwen_prompt(path: Path) -> object:
  """Load a safe Qwen prompt and reconstruct upstream prompt item objects."""
  from app.backends.pytorch.qwen_runtime import VoiceClonePromptItem

  json_path = path.with_suffix(".json")
  tensor_path = path.with_suffix(".safetensors")
  metadata = json.loads(json_path.read_text(encoding="utf-8"))
  if metadata.get("schema_version") != 1:
    raise ValueError("Unsupported prompt cache schema.")
  items = metadata.get("items")
  if not isinstance(items, list) or not items:
    raise ValueError("Prompt cache contains no items.")
  tensors = load_file(str(tensor_path), device="cpu")
  prompt_items = []
  for index, item in enumerate(items):
    if not isinstance(item, dict):
      raise ValueError("Prompt cache item metadata is invalid.")
    code_key = f"item_{index}_code"
    embedding_key = f"item_{index}_embedding"
    if embedding_key not in tensors:
      raise ValueError("Prompt cache is missing a speaker embedding.")
    has_reference_code = item.get("has_ref_code") is True
    if has_reference_code and code_key not in tensors:
      raise ValueError("Prompt cache is missing reference codes.")
    prompt_items.append(
      VoiceClonePromptItem(
        ref_code=tensors.get(code_key) if has_reference_code else None,
        ref_spk_embedding=tensors[embedding_key],
        x_vector_only_mode=bool(item.get("x_vector_only_mode")),
        icl_mode=bool(item.get("icl_mode")),
        ref_text=item.get("ref_text") if isinstance(item.get("ref_text"), str) else None,
      )
    )
  return prompt_items
