/** Shared model-selection behavior tests. */

import { describe, expect, it } from "vitest";

import { preferredModel } from "./AppContext";

const models = [
  {
    backend_supported: true,
    id: "qwen3-asr-1.7B",
    kind: "transcription",
    load_state: "unloaded",
  },
  {
    backend_supported: true,
    id: "nvidia-canary-1b-v2",
    kind: "transcription",
    load_state: "loaded",
  },
  {
    backend_supported: true,
    id: "qwen-tts-1.7B",
    kind: "tts",
    load_state: "loaded",
  },
];

describe("shared model selection", () => {
  it("promotes the loaded transcription model over an old selection", () => {
    expect(preferredModel(models, "transcription", "qwen3-asr-1.7B")).toBe(
      "nvidia-canary-1b-v2",
    );
  });
});
