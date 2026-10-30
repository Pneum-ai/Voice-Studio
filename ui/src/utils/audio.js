/** Browser audio helpers for streaming PCM and constructing WAV blobs. */

/** Write one ASCII header token into a byte view. */
function writeAscii(view, offset, value) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

/** Combine signed 16-bit PCM chunks into a downloadable WAV blob. */
export function pcmChunksToWav(chunks, sampleRate, channels = 1) {
  const dataLength = chunks.reduce((sum, chunk) => sum + chunk.byteLength, 0);
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  chunks.forEach((chunk) => {
    new Uint8Array(buffer, offset, chunk.byteLength).set(new Uint8Array(chunk));
    offset += chunk.byteLength;
  });
  return new Blob([buffer], { type: "audio/wav" });
}

/** Schedule raw PCM chunks for gap-free Web Audio playback. */
export class PcmStreamPlayer {
  /** Initialize a player without allocating browser audio resources. */
  constructor() {
    this.context = null;
    this.nextStart = 0;
    this.sources = new Set();
    this.version = 0;
  }

  /** Create or resume the audio context after a user gesture. */
  async start() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error("This browser does not support live audio playback.");
    }
    this.stop();
    const version = this.version;
    if (!this.context || this.context.state === "closed") {
      this.context = new AudioContextClass();
    }
    const context = this.context;
    let timeout;
    try {
      // Invoke resume synchronously in Generate's user gesture, before any network wait.
      await Promise.race([
        context.resume(),
        new Promise((_, reject) => {
          timeout = window.setTimeout(() => reject(new Error(
            "Live audio could not start. Allow sound for this site, then try Generate again.",
          )), 5_000);
        }),
      ]);
      if (this.context !== context || version !== this.version) {
        throw new DOMException("Playback cancelled.", "AbortError");
      }
      if (context.state !== "running") {
        throw new Error("Live audio is paused. Allow sound for this site and try again.");
      }
      this.nextStart = context.currentTime;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  /** Decode and schedule the next PCM frame. */
  push(arrayBuffer, sampleRate, channels = 1) {
    if (!this.context || this.context.state !== "running") {
      throw new Error("Live audio was interrupted. Check your audio output and try again.");
    }
    const samples = new Int16Array(arrayBuffer);
    const frames = Math.floor(samples.length / channels);
    const audioBuffer = this.context.createBuffer(channels, frames, sampleRate);
    for (let channel = 0; channel < channels; channel += 1) {
      const output = audioBuffer.getChannelData(channel);
      for (let frame = 0; frame < frames; frame += 1) {
        output[frame] = samples[frame * channels + channel] / 32768;
      }
    }
    const source = this.context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.context.destination);
    const startAt = Math.max(this.nextStart, this.context.currentTime);
    source.start(startAt);
    this.nextStart = startAt + audioBuffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
  }

  /** Stop scheduled sources while retaining an unlocked context for the next request. */
  stop() {
    this.version += 1;
    this.sources.forEach((source) => {
      try {
        source.stop();
      } catch {
        // A source that has already ended needs no cleanup.
      }
    });
    this.sources.clear();
    this.nextStart = 0;
  }

  /** Release browser audio resources when leaving the speech workspace. */
  close() {
    this.stop();
    const context = this.context;
    this.context = null;
    if (context && context.state !== "closed") {
      void context.close().catch(() => {});
    }
  }
}
