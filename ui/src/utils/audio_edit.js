/** Browser-side audio decoding for waveform and trim-coordinate selection. */

/** Decode an uploaded audio file once for waveform and trim operations. */
export async function decodeAudioFile(file) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("This browser does not support audio editing.");
  }
  const context = new AudioContextClass();
  try {
    const encoded = await file.arrayBuffer();
    return await context.decodeAudioData(encoded.slice(0));
  } finally {
    await context.close();
  }
}
