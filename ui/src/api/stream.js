/** Persistent WebSocket transport for low-latency PCM speech. */

import { PcmStreamPlayer, pcmChunksToWav } from "../utils/audio";

/** Manage one reusable speech WebSocket and its active request. */
export class SpeechStreamClient {
  /** Initialize disconnected transport and request state. */
  constructor() {
    this.socket = null;
    this.openPromise = null;
    this.pending = null;
    this.keepalive = null;
    this.requestCounter = 0;
    this.player = new PcmStreamPlayer();
  }

  /** Open the socket once and reuse it for sequential generations. */
  async connect() {
    if (this.socket?.readyState === WebSocket.OPEN) {
      return;
    }
    if (this.openPromise) {
      return this.openPromise;
    }
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/v1/speech/stream`;
    const socket = new WebSocket(url);
    this.socket = socket;
    socket.binaryType = "arraybuffer";
    socket.addEventListener("message", (event) => {
      if (this.socket === socket) this.handleMessage(event);
    });
    socket.addEventListener("close", () => {
      if (this.socket === socket) this.handleClose();
    });
    this.openPromise = new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", () => reject(new Error(
        "The speech stream could not connect.",
      )), { once: true });
      socket.addEventListener("close", () => reject(new Error(
        "The speech stream disconnected before it opened.",
      )), { once: true });
    }).finally(() => {
      this.openPromise = null;
    });
    await this.openPromise;
    if (this.socket !== socket) return;
    this.keepalive = window.setInterval(() => {
      if (this.socket?.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ type: "ping" }));
      }
    }, 20_000);
  }

  /** Generate, play, and assemble one streamed speech request. */
  generate(payload, callbacks = {}) {
    if (this.pending) {
      return Promise.reject(new Error("A streamed generation is already active."));
    }
    // IDs correlate requests within this client; they do not need a secure-context API.
    const requestId = `speech-${++this.requestCounter}`;
    return new Promise((resolve, reject) => {
      const pending = {
        requestId,
        chunks: [],
        sampleRate: 24_000,
        channels: 1,
        resolve,
        reject,
        callbacks,
        sent: false,
      };
      this.pending = pending;
      // start() must be called before the first await so the browser sees the click.
      Promise.all([this.player.start(), this.connect()]).then(() => {
        if (this.pending !== pending) {
          return;
        }
        this.socket.send(JSON.stringify({
          ...payload,
          type: "generate",
          request_id: requestId,
        }));
        pending.sent = true;
      }).catch((error) => {
        if (this.pending === pending) {
          this.fail(error);
        }
      });
    });
  }

  /** Cancel the active request without closing the reusable socket. */
  cancel() {
    if (!this.pending) {
      return;
    }
    this.player.stop();
    if (!this.pending.sent) {
      this.fail(new DOMException("Generation cancelled.", "AbortError"));
      return;
    }
    this.socket.send(JSON.stringify({
      type: "cancel",
      request_id: this.pending.requestId,
    }));
  }

  /** Route one control or binary server message. */
  handleMessage(event) {
    if (typeof event.data !== "string") {
      this.handleAudio(event.data);
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === "pong") {
      return;
    }
    const pending = this.pending;
    if (!pending || message.request_id !== pending.requestId) {
      return;
    }
    if (message.type === "started") {
      pending.sampleRate = message.sample_rate;
      pending.channels = message.channels;
      pending.callbacks.onStarted?.(message);
      return;
    }
    if (message.type === "finished") {
      const blob = pcmChunksToWav(
        pending.chunks,
        pending.sampleRate,
        pending.channels,
      );
      pending.resolve({ blob, generation: message.generation });
      this.pending = null;
      return;
    }
    if (message.type === "cancelled") {
      this.player.stop();
      pending.reject(new DOMException("Generation cancelled.", "AbortError"));
      this.pending = null;
      return;
    }
    if (message.type === "error") {
      const error = new Error(message.detail || message.title);
      error.code = message.code;
      this.fail(error);
    }
  }

  /** Play and retain one raw PCM frame. */
  handleAudio(arrayBuffer) {
    if (!this.pending) {
      return;
    }
    try {
      this.player.push(arrayBuffer, this.pending.sampleRate, this.pending.channels);
      this.pending.chunks.push(arrayBuffer);
      this.pending.callbacks.onChunk?.(arrayBuffer.byteLength);
    } catch (error) {
      this.fail(error);
      this.socket?.close();
    }
  }

  /** Surface playback/transport failures to the UI and stop queued audio. */
  fail(error) {
    const pending = this.pending;
    this.pending = null;
    if (pending) this.player.stop();
    pending?.reject(error);
  }

  /** Reject active work when the socket closes unexpectedly. */
  handleClose() {
    if (this.keepalive) {
      window.clearInterval(this.keepalive);
      this.keepalive = null;
    }
    this.fail(new Error("The speech stream disconnected."));
    this.socket = null;
  }

  /** Release the socket, audio player, and active request. */
  close() {
    this.fail(new DOMException("Playback cancelled.", "AbortError"));
    this.player.close();
    window.clearInterval(this.keepalive);
    this.keepalive = null;
    this.socket?.close();
    this.socket = null;
  }
}
