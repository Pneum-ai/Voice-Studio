/** Small browser PCM recorder that produces a backend-friendly WAV blob. */

/** Encode mono float samples as a 16-bit PCM WAV file. */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeText = (offset, value) => {
    [...value].forEach((character, index) => {
      view.setUint8(offset + index, character.charCodeAt(0));
    });
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const clipped = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + index * 2, clipped * (clipped < 0 ? 32768 : 32767), true);
  });
  return new Blob([buffer], { type: "audio/wav" });
}

/** Merge sequential audio callback chunks into one typed array. */
function mergeChunks(chunks) {
  const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

/** Start microphone capture and return an explicit asynchronous stop function. */
export async function startWavRecording({ onLevel } = {}) {
  if (!navigator.mediaDevices?.getUserMedia || !globalThis.AudioContext) {
    throw new Error("Microphone recording needs a supported browser on HTTPS or localhost.");
  }
  // Avoid browser gain pumping that can drive a reference into saturation.
  // These preferences may be ignored by devices that do not support them.
  const stream = await navigator.mediaDevices.getUserMedia({ audio: {
    channelCount: 1,
    autoGainControl: false,
    echoCancellation: false,
    noiseSuppression: false,
  } });
  let context;
  let source;
  let processor;
  const chunks = [];
  const release = async () => {
    if (processor) {
      processor.onaudioprocess = null;
      processor.disconnect();
    }
    source?.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    if (context && context.state !== "closed") {
      await context.close();
    }
  };
  try {
    context = new AudioContext();
    source = context.createMediaStreamSource(stream);
    processor = context.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => {
      const samples = new Float32Array(event.inputBuffer.getChannelData(0));
      chunks.push(samples);
      let peak = 0;
      let power = 0;
      for (const sample of samples) {
        peak = Math.max(peak, Math.abs(sample));
        power += sample * sample;
      }
      onLevel?.({ peak, rms: Math.sqrt(power / Math.max(1, samples.length)) });
    };
    source.connect(processor);
    processor.connect(context.destination);
    await context.resume();
  } catch (error) {
    await release();
    throw error;
  }
  let stopped;
  return () => {
    stopped ??= (async () => {
      const sampleRate = context.sampleRate;
      await release();
      return encodeWav(mergeChunks(chunks), sampleRate);
    })();
    return stopped;
  };
}
