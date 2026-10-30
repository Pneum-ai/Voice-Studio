/** PCM capture contracts and microphone cleanup when initialization fails. */

import { afterEach, expect, it, vi } from "vitest";

import { startWavRecording } from "./recording";

/** Install browser audio substitutes at the hardware boundary. */
function microphone({ resumeError } = {}) {
  const track = { stop: vi.fn() };
  const source = { connect: vi.fn(), disconnect: vi.fn() };
  const processor = { connect: vi.fn(), disconnect: vi.fn(), onaudioprocess: null };
  const context = {
    sampleRate: 16000,
    state: "running",
    createMediaStreamSource: () => source,
    createScriptProcessor: () => processor,
    resume: resumeError ? vi.fn().mockRejectedValue(resumeError) : vi.fn().mockResolvedValue(),
    close: vi.fn().mockResolvedValue(),
  };
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [track] }) },
  });
  vi.stubGlobal("AudioContext", class { constructor() { return context; } });
  return { track, context, processor };
}

afterEach(() => vi.unstubAllGlobals());

it("encodes captured samples as mono PCM WAV and stops capture only once", async () => {
  const { track, context, processor } = microphone();
  const onLevel = vi.fn();
  const stop = await startWavRecording({ onLevel });
  processor.onaudioprocess({
    inputBuffer: { getChannelData: () => new Float32Array([-1, 0, 1]) },
  });
  expect(onLevel).toHaveBeenCalledWith({ peak: 1, rms: Math.sqrt(2 / 3) });
  const blob = await stop();
  expect(await stop()).toBe(blob);
  const bytes = await new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsArrayBuffer(blob);
  });
  const view = new DataView(bytes);
  expect(String.fromCharCode(...new Uint8Array(bytes, 0, 4))).toBe("RIFF");
  expect(view.getUint32(24, true)).toBe(16000);
  expect(view.getUint32(40, true)).toBe(6);
  expect([44, 46, 48].map((offset) => view.getInt16(offset, true))).toEqual([-32768, 0, 32767]);
  expect(track.stop).toHaveBeenCalledTimes(1);
  expect(context.close).toHaveBeenCalledTimes(1);
  expect(processor.onaudioprocess).toBeNull();
});

it("releases the microphone if the audio context cannot start", async () => {
  const { track, context } = microphone({ resumeError: new Error("Audio unavailable") });
  await expect(startWavRecording()).rejects.toThrow("Audio unavailable");
  expect(track.stop).toHaveBeenCalledTimes(1);
  expect(context.close).toHaveBeenCalledTimes(1);
});
