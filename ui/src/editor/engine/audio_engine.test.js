/** Stereo WAV export format regression. */

import { describe, expect, it } from "vitest";

import { encodeWav } from "./audio_engine";

describe("editor audio engine", () => {
  it("encodes stereo PCM with a valid WAV header", async () => {
    const buffer = {
      length: 2,
      numberOfChannels: 2,
      sampleRate: 48000,
      getChannelData: (channel) => (
        channel ? new Float32Array([-1, 0]) : new Float32Array([1, 0])
      ),
    };
    const blob = encodeWav(buffer);
    const view = new DataView(await blob.arrayBuffer());
    const text = (offset, length) => String.fromCharCode(
      ...new Uint8Array(view.buffer, offset, length),
    );

    expect(text(0, 4)).toBe("RIFF");
    expect(text(8, 4)).toBe("WAVE");
    expect(view.getUint16(22, true)).toBe(2);
    expect(view.getUint32(24, true)).toBe(48000);
    expect(blob.size).toBe(52);
  });
});
