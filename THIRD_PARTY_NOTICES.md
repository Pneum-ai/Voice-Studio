# Third-Party Notices

This project integrates with third-party libraries and model weights. Their
licenses remain independent from this repository's license.

## Qwen3-TTS

The PyTorch backend uses the `qwen-tts` package and Qwen3-TTS model
repositories published by the Alibaba Qwen team. The installed upstream
wrapper identifies itself as Apache-2.0 licensed.

Project: https://github.com/QwenLM/Qwen3-TTS

Users must review the license and acceptable-use terms shipped with the exact
model revision they download.

## Nari Qwen3-TTS

The `nari_qwen3_tts` package bundled in Voice Studio releases is built from
`vendor/nari-qwen3-tts`, which is derived from
[`nari-labs/nari-qwen3-tts`](https://github.com/nari-labs/nari-qwen3-tts) at commit
`e8c5b2bf6d65a965037f161d713bfa3e8856feb3`. It is distributed under the Apache License
2.0; the upstream license is retained at `vendor/nari-qwen3-tts/LICENSE`.

Voice Studio modifies this engine to support Qwen3-TTS 0.6B and 1.7B Base checkpoints,
including continuous prefill embeddings and hidden reference-Codec context.

## Whisper

The transcription backend uses OpenAI Whisper model repositories through
Transformers or `mlx-audio`.

Project: https://github.com/openai/whisper

Users must review the license and model documentation for the exact revision
they download.

## Qwen3-ASR

The optional transcription worker uses the `qwen-asr` package and Qwen3-ASR
model repositories published by the Alibaba Qwen team under Apache-2.0 terms.

Project: https://github.com/QwenLM/Qwen3-ASR

## NVIDIA Canary and Parakeet

The optional NVIDIA transcription adapter uses NeMo with Canary and Parakeet
model repositories. The cataloged Canary 1B v2 and Parakeet TDT 0.6B v3 model
cards identify their governing model terms as CC-BY-4.0.

Projects: https://github.com/NVIDIA-NeMo/NeMo and https://huggingface.co/nvidia

Users and distributors must review the exact checkpoint, dataset, and runtime
terms before downloading or redistributing model artifacts.

## MLX Audio

The optional Apple Silicon backend uses `mlx-audio` and converted model
repositories.

Project: https://github.com/Blaizzy/mlx-audio

The serving modules `app/backends/mlx/session.py`, `app/backends/mlx/prompts.py`,
and `app/backends/mlx/codec.py`
adapt Qwen generation, cache handling, and decoder math from mlx-audio 0.5.9. Its upstream MIT notice follows:

```text
MIT License

Copyright (c) 2024 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Converted weights may carry additional upstream notices. Review repository
metadata before redistribution.

## Other Dependencies

The dependency lockfile records the exact resolved Python packages for this
checkout. Distributors are responsible for reproducing a complete notice and
license inventory for their release artifact.
