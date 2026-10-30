/** Web Audio decoding, scheduled preview, offline rendering, and WAV export. */

import { clipDuration, projectDuration } from "../state/project";

/** Decode an audio blob or authenticated URL into a reusable AudioBuffer. */
export async function decodeEditorAudio(input) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("This browser does not support audio editing.");
  }
  const blob = input instanceof Blob
    ? input
    : await fetch(input).then((response) => {
      if (!response.ok) {
        throw new Error(`Audio request failed (${response.status}).`);
      }
      return response.blob();
    });
  const context = new AudioContextClass();
  try {
    return await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    await context.close();
  }
}

/** Reduce an AudioBuffer to stable waveform bar heights. */
export function waveformPeaks(buffer, count = 80) {
  const samples = buffer.getChannelData(0);
  return Array.from({ length: count }, (_, index) => {
    const start = Math.floor((index * samples.length) / count);
    const end = Math.max(start + 1, Math.floor(((index + 1) * samples.length) / count));
    let peak = 0;
    for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
      peak = Math.max(peak, Math.abs(samples[sampleIndex] || 0));
    }
    return Math.max(8, Math.round(peak * 100));
  });
}

/** Return gain at a local clip time after fades are applied. */
function clipGainAt(clip, localTime) {
  const duration = clipDuration(clip);
  const fadeInGain = clip.fadeIn ? Math.min(1, localTime / clip.fadeIn) : 1;
  const remaining = duration - localTime;
  const fadeOutGain = clip.fadeOut ? Math.min(1, remaining / clip.fadeOut) : 1;
  return clip.gain * Math.max(0, Math.min(fadeInGain, fadeOutGain));
}

/** Schedule one trimmed clip with its gain and fade envelope. */
function scheduleClip(context, output, buffer, clip, offset) {
  const audibleStart = Math.max(clip.start, offset);
  const localStart = audibleStart - clip.start;
  const remaining = clipDuration(clip) - localStart;
  if (remaining <= 0) {
    return null;
  }
  const when = context.currentTime + audibleStart - offset;
  const source = context.createBufferSource();
  const gain = context.createGain();
  source.buffer = buffer;
  source.connect(gain).connect(output);
  gain.gain.setValueAtTime(clipGainAt(clip, localStart), when);
  if (clip.fadeIn > localStart && clip.fadeIn < clipDuration(clip)) {
    gain.gain.linearRampToValueAtTime(clip.gain, when + clip.fadeIn - localStart);
  }
  const fadeOutStart = clipDuration(clip) - clip.fadeOut;
  if (clip.fadeOut && fadeOutStart >= localStart) {
    gain.gain.setValueAtTime(clip.gain, when + fadeOutStart - localStart);
    gain.gain.linearRampToValueAtTime(0, when + remaining);
  }
  source.start(when, clip.inPoint + localStart, remaining);
  return source;
}

/** Start a scheduled project preview and return an explicit stop function. */
export function previewProject(project, buffers, offset = 0, onEnded = () => {}) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  const sources = project.clips.flatMap((clip) => {
    const source = scheduleClip(
      context,
      context.destination,
      buffers.get(clip.sourceId),
      clip,
      offset,
    );
    return source ? [source] : [];
  });
  const remaining = Math.max(0, projectDuration(project) - offset);
  const timer = window.setTimeout(onEnded, remaining * 1000 + 40);
  return async () => {
    window.clearTimeout(timer);
    sources.forEach((source) => {
      try {
        source.stop();
      } catch {
        // A source that naturally ended no longer needs stopping.
      }
    });
    await context.close();
  };
}

/** Render the project into a stereo AudioBuffer without real-time playback. */
export async function renderProject(project, buffers) {
  const duration = Math.max(0.1, projectDuration(project));
  const sampleRate = 48000;
  const OfflineContextClass = window.OfflineAudioContext
    || window.webkitOfflineAudioContext;
  if (!OfflineContextClass) {
    throw new Error("This browser does not support offline audio export.");
  }
  const context = new OfflineContextClass(
    2,
    Math.ceil(duration * sampleRate),
    sampleRate,
  );
  project.clips.forEach((clip) => {
    scheduleClip(context, context.destination, buffers.get(clip.sourceId), clip, 0);
  });
  return context.startRendering();
}

/** Encode an AudioBuffer as interleaved 16-bit PCM WAV. */
export function encodeWav(buffer) {
  const channels = Math.min(2, buffer.numberOfChannels);
  const frameCount = buffer.length;
  const output = new ArrayBuffer(44 + frameCount * channels * 2);
  const view = new DataView(output);
  const writeText = (offset, value) => [...value].forEach((character, index) => {
    view.setUint8(offset + index, character.charCodeAt(0));
  });
  writeText(0, "RIFF");
  view.setUint32(4, output.byteLength - 8, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, frameCount * channels * 2, true);
  const channelData = Array.from(
    { length: channels },
    (_, channel) => buffer.getChannelData(channel),
  );
  let offset = 44;
  for (let frame = 0; frame < frameCount; frame += 1) {
    for (let channel = 0; channel < channels; channel += 1) {
      const sample = Math.max(-1, Math.min(1, channelData[channel][frame]));
      view.setInt16(offset, sample * (sample < 0 ? 32768 : 32767), true);
      offset += 2;
    }
  }
  return new Blob([output], { type: "audio/wav" });
}

/** Render and download one project with a filesystem-safe name. */
export async function exportProject(project, buffers) {
  const blob = encodeWav(await renderProject(project, buffers));
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  const safeName = project.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  link.download = `${safeName || "voice-scene"}.wav`;
  link.click();
  URL.revokeObjectURL(url);
}
