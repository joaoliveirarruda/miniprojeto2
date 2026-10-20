export const EMOTIONS_IN_TEST_ORDER = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"];

/** Minimal 16-bit PCM mono WAV reader, enough for the fixtures. */
export function readWav(buf: Buffer): { audio: Float32Array; sampleRate: number } {
  const sampleRate = buf.readUInt32LE(24);
  let offset = 12;
  while (buf.toString("ascii", offset, offset + 4) !== "data") offset += 8 + buf.readUInt32LE(offset + 4);
  const size = buf.readUInt32LE(offset + 4);
  const pcm = new Int16Array(buf.buffer.slice(buf.byteOffset + offset + 8, buf.byteOffset + offset + 8 + size));
  return { audio: Float32Array.from(pcm, (s) => s / 32768), sampleRate };
}
