/** Persistent WebSocket framing, reuse, and PCM assembly tests. */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SpeechStreamClient } from "./stream";

/** Minimal Web Audio context used by transport tests. */
class FakeAudioContext {
  /** Initialize a running deterministic audio clock. */
  constructor() {
    this.currentTime = 0;
    this.state = "running";
    this.destination = {};
  }

  /** Satisfy the browser resume contract. */
  async resume() {}

  /** Release the fake device on workspace cleanup. */
  async close() { this.state = "closed"; }

  /** Allocate a mutable mono audio buffer stub. */
  createBuffer(_channels, frames, sampleRate) {
    return {
      duration: frames / sampleRate,
      getChannelData: () => new Float32Array(frames),
    };
  }

  /** Build a schedulable buffer source stub. */
  createBufferSource() {
    return {
      connect: vi.fn(),
      start: vi.fn(),
      stop: vi.fn(),
      onended: null,
    };
  }
}

/** Deterministic event-driven WebSocket test double. */
class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSED = 3;

  /** Initialize an opening socket and retain the latest instance. */
  constructor() {
    this.readyState = FakeWebSocket.CONNECTING;
    this.listeners = new Map();
    this.sent = [];
    FakeWebSocket.instance = this;
    queueMicrotask(() => {
      this.readyState = FakeWebSocket.OPEN;
      this.emit("open", {});
    });
  }

  /** Register a transport event listener. */
  addEventListener(name, handler) {
    const handlers = this.listeners.get(name) || [];
    handlers.push(handler);
    this.listeners.set(name, handlers);
  }

  /** Capture one outbound JSON message. */
  send(payload) {
    this.sent.push(JSON.parse(payload));
  }

  /** Deliver one synthetic socket event. */
  emit(name, event) {
    (this.listeners.get(name) || []).forEach((handler) => handler(event));
  }

  /** Close the socket and deliver its close event. */
  close() {
    this.readyState = FakeWebSocket.CLOSED;
    this.emit("close", {});
  }
}

describe("speech stream", () => {
  beforeEach(() => {
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.stubGlobal("AudioContext", FakeAudioContext);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("sends a generation, consumes PCM, and builds a playable WAV", async () => {
    const sources = vi.spyOn(FakeAudioContext.prototype, "createBufferSource");
    const client = new SpeechStreamClient();
    const resultPromise = client.generate({
      profile_id: "profile-1",
      text: "Hello.",
      language: "en",
      model: "qwen-tts-1.7B",
      store: false,
    });
    await vi.waitFor(() => expect(FakeWebSocket.instance.sent).toHaveLength(1));
    const request = FakeWebSocket.instance.sent[0];
    expect(request.type).toBe("generate");

    FakeWebSocket.instance.emit("message", {
      data: JSON.stringify({
        type: "started",
        request_id: request.request_id,
        sample_rate: 24_000,
        channels: 1,
      }),
    });
    FakeWebSocket.instance.emit("message", {
      data: new Int16Array([0, 100, -100, 0]).buffer,
    });
    // Live PCM must be scheduled before the server sends its completion message.
    expect(sources.mock.results[0].value.start).toHaveBeenCalledOnce();
    FakeWebSocket.instance.emit("message", {
      data: JSON.stringify({
        type: "finished",
        request_id: request.request_id,
        generation: null,
      }),
    });

    const result = await resultPromise;
    expect(result.blob.type).toBe("audio/wav");
    expect(result.blob.size).toBe(52);
    client.close();
  });

  it("sends cancellation for the active request", async () => {
    const client = new SpeechStreamClient();
    const resultPromise = client.generate({
      profile_id: "profile-1",
      text: "Cancel me.",
      language: "en",
      model: "qwen-tts-1.7B",
      store: false,
    });
    await vi.waitFor(() => expect(FakeWebSocket.instance.sent).toHaveLength(1));
    const request = FakeWebSocket.instance.sent[0];
    client.cancel();
    expect(FakeWebSocket.instance.sent[1]).toEqual({
      type: "cancel",
      request_id: request.request_id,
    });
    FakeWebSocket.instance.emit("message", {
      data: JSON.stringify({
        type: "cancelled",
        request_id: request.request_id,
      }),
    });
    await expect(resultPromise).rejects.toMatchObject({ name: "AbortError" });
    client.close();
  });

  it("unlocks audio in the initiating gesture before waiting for the socket", async () => {
    let gestureActive = true;
    const resume = vi.spyOn(FakeAudioContext.prototype, "resume")
      .mockImplementation(() => {
        if (!gestureActive) {
          return Promise.reject(new Error("Playback requires a user gesture."));
        }
        return Promise.resolve();
      });
    const client = new SpeechStreamClient();
    const result = client.generate({ text: "Play immediately." }).catch((error) => error);
    gestureActive = false;
    try {
      expect(resume).toHaveBeenCalledOnce();
      await vi.waitFor(() => expect(FakeWebSocket.instance.sent).toHaveLength(1));
    } finally {
      client.close();
      await result;
      resume.mockRestore();
    }
  });

  it("reports blocked audio instead of hanging or sending a silent request", async () => {
    vi.useFakeTimers();
    vi.spyOn(FakeAudioContext.prototype, "resume").mockReturnValue(new Promise(() => {}));
    const client = new SpeechStreamClient();
    const result = client.generate({ text: "Blocked audio." }).catch((error) => error);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(await result).toMatchObject({ message: expect.stringContaining("Allow sound") });
    expect(FakeWebSocket.instance.sent).toEqual([]);
    expect(client.pending).toBeNull();
    client.close();
  });

  it("cancels during audio startup without dispatching generation later", async () => {
    let resume;
    vi.spyOn(FakeAudioContext.prototype, "resume")
      .mockReturnValue(new Promise((resolve) => { resume = resolve; }));
    const client = new SpeechStreamClient();
    const result = client.generate({ text: "Cancel before startup." }).catch((error) => error);
    client.cancel();
    expect(await result).toMatchObject({ name: "AbortError" });
    resume();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(FakeWebSocket.instance.sent).toEqual([]);
    client.close();
  });

  it("surfaces a PCM playback failure rather than reporting successful generation", async () => {
    const client = new SpeechStreamClient();
    const result = client.generate({ text: "Interrupted audio." }).catch((error) => error);
    await vi.waitFor(() => expect(FakeWebSocket.instance.sent).toHaveLength(1));
    client.player.context.state = "suspended";
    FakeWebSocket.instance.emit("message", { data: new Int16Array([1, 2]).buffer });
    expect(await result).toMatchObject({ message: expect.stringContaining("interrupted") });
    expect(FakeWebSocket.instance.readyState).toBe(FakeWebSocket.CLOSED);
    client.close();
  });
});
