"""In-memory optimizations for the single MLX Qwen serving path."""

from __future__ import annotations

from typing import Any


def optimize_model(model: Any) -> None:
  """Quantize Talker weights once and compile residual prediction for its sampling settings."""
  import mlx.core as mx  # pyright: ignore[reportMissingImports]
  import mlx.nn as nn  # pyright: ignore[reportMissingImports]

  nn.quantize(model.talker, group_size=64, bits=8)
  mx.eval(model.talker.parameters())
  original = model._predict_code_tokens
  compiled: dict[tuple[float, int, float], Any] = {}

  def replacement(first: Any, hidden: Any, **kwargs: Any) -> Any:
    """Cache graphs by sampling configuration and avoid prefill-length-dependent shapes."""
    key = (kwargs["temperature"], kwargs["top_k"], kwargs["top_p"])
    if key not in compiled:

      def predict(token: Any, state: Any) -> Any:
        """Trace residual prediction with explicit random-state capture."""
        return original(token, state, temperature=key[0], top_k=key[1], top_p=key[2])

      compiled[key] = mx.compile(predict, inputs=mx.random.state, outputs=mx.random.state)
    return compiled[key](first, hidden[:, -1:, :])

  model._predict_code_tokens = replacement
