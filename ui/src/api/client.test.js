/** Multipart upload serialization and binary speech response regressions. */

import { afterEach, expect, it, vi } from "vitest";

import { api } from "./client";

afterEach(() => vi.restoreAllMocks());

it("uploads the original audio with trim coordinates for samples and transcription", async () => {
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => (
    new Response(JSON.stringify({ ok: true }))
  ));
  const audio = new File(["audio"], "sample.wav", { type: "audio/wav" });
  const trim = { endSeconds: 8.5, startSeconds: 1.25 };

  await api.addSample("profile-1", audio, "Exact words.", trim);
  await api.transcribe(audio, "whisper-base", "en", "Pneum", trim);

  const [sample, transcription] = fetchMock.mock.calls.map(([, options]) => options.body);
  for (const body of [sample, transcription]) {
    expect(body.get("audio")).toBe(audio);
    expect(body.get("trim_start_seconds")).toBe("1.25");
    expect(body.get("trim_end_seconds")).toBe("8.5");
  }
  expect(sample.get("transcript")).toBe("Exact words.");
  expect(transcription.get("model")).toBe("whisper-base");
  expect(transcription.get("language")).toBe("en");
  expect(transcription.get("context")).toBe("Pneum");
});

it("preserves the speech response as an audio blob", async () => {
  // Node's Response requires stream(), which jsdom's Blob does not provide.
  vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(
    new Uint8Array([119, 97, 118]), { headers: { "Content-Type": "audio/wav" } },
  ));
  const audio = await api.generateSpeech({ profile_id: "profile-1", text: "Hello." });
  expect(audio.type).toBe("audio/wav");
  expect(await audio.text()).toBe("wav");
});
