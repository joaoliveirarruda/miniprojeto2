import type { Score } from "./inference";
import { SAMPLE_RATE } from "./inference";

export const WINDOW_SECONDS = 3;
// ponytail: fixed RMS gate; mics differ in gain, so this is the knob to turn if the
// app says "waiting for speech" while you talk (lower) or reacts to room noise (raise).
export const SILENCE_RMS = 0.004; // RAVDESS "calm" speech has a median RMS of ~0.005

export const rms = (x: Float32Array) => Math.sqrt(x.reduce((a, v) => a + v * v, 0) / Math.max(1, x.length));

export async function resampleTo16k(samples: Float32Array, rate: number): Promise<Float32Array> {
  if (rate === SAMPLE_RATE) return samples;
  const ctx = new OfflineAudioContext(1, Math.ceil((samples.length * SAMPLE_RATE) / rate), SAMPLE_RATE);
  const buffer = ctx.createBuffer(1, samples.length, rate);
  buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

/** Any browser-decodable audio file -> mono 16 kHz samples. */
export async function decodeTo16k(data: ArrayBuffer): Promise<Float32Array> {
  // An OfflineAudioContext decodes straight to its own sample rate.
  const buffer = await new OfflineAudioContext(1, 1, SAMPLE_RATE).decodeAudioData(data);
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    buffer.getChannelData(c).forEach((v, i) => (mono[i] += v / buffer.numberOfChannels));
  }
  return mono;
}

/** 3 s windows every 1.5 s; a clip shorter than a window is one window. */
export function windows(x: Float32Array, size = WINDOW_SECONDS * SAMPLE_RATE, hop = size / 2) {
  if (x.length <= size) return [{ start: 0, audio: x }];
  const out = [];
  for (let s = 0; s + size <= x.length; s += hop) out.push({ start: s / SAMPLE_RATE, audio: x.subarray(s, s + size) });
  return out;
}

/** Exponential moving average over score lists, so the live display doesn't flicker. */
export function smooth(prev: Score[] | null, next: Score[], alpha = 0.4): Score[] {
  if (!prev) return next;
  return next.map((s, i) => ({ label: s.label, score: alpha * s.score + (1 - alpha) * prev[i].score }));
}

export function mean(lists: Score[][]): Score[] {
  return lists[0].map((s, i) => ({ label: s.label, score: lists.reduce((a, l) => a + l[i].score, 0) / lists.length }));
}

export const top = (scores: Score[]) => scores.reduce((a, b) => (b.score > a.score ? b : a));

/** Keeps the last `size` samples pushed into it. */
export class RingBuffer {
  private data: Float32Array;
  private filled = 0;
  constructor(size: number) {
    this.data = new Float32Array(size);
  }
  push(chunk: Float32Array) {
    const n = Math.min(chunk.length, this.data.length);
    this.data.copyWithin(0, n);
    this.data.set(chunk.subarray(chunk.length - n), this.data.length - n);
    this.filled = Math.min(this.data.length, this.filled + n);
  }
  /** The buffered audio, oldest first (shorter than `size` until it fills up). */
  read() {
    return this.data.slice(this.data.length - this.filled);
  }
}
