/** Decode browser audio into normalized waveform peaks for visual scrubbing. */

/** Build a deterministic placeholder while decoded peaks are unavailable. */
export function fallbackWaveformBars(count) {
  return Array.from({ length: count }, (_, index) => {
    const wave = Math.abs(Math.sin(index * 0.71) * Math.cos(index * 0.19));
    return 18 + Math.round(wave * 78);
  });
}

/** Reduce one channel of audio samples into normalized percentage peaks. */
export function summarizeWaveform(samples, count) {
  if (!samples.length || count < 1) {
    return [];
  }
  const blockSize = Math.max(1, Math.ceil(samples.length / count));
  const peaks = Array.from({ length: count }, (_, index) => {
    const start = index * blockSize;
    const end = Math.min(samples.length, start + blockSize);
    let peak = 0;
    for (let position = start; position < end; position += 1) {
      peak = Math.max(peak, Math.abs(samples[position]));
    }
    return peak;
  });
  const maximum = Math.max(...peaks, Number.EPSILON);
  return peaks.map((peak) => 16 + Math.round((peak / maximum) * 84));
}

/** Fetch and decode an audio URL into waveform bar heights. */
export async function loadWaveformBars(src, count, signal) {
  const response = await fetch(src, { signal });
  if (!response.ok) {
    throw new Error(`Waveform audio request failed (${response.status}).`);
  }
  const encoded = await response.arrayBuffer();
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    return fallbackWaveformBars(count);
  }
  const context = new AudioContextClass();
  try {
    const decoded = await context.decodeAudioData(encoded.slice(0));
    return summarizeWaveform(decoded.getChannelData(0), count);
  } finally {
    await context.close();
  }
}
