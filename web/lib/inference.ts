import { env, pipeline, type AudioClassificationPipeline } from "@huggingface/transformers";

export const SAMPLE_RATE = 16000;
const MIN_SAMPLES = 400; // wav2vec2's conv front-end needs at least one 25 ms frame

export type InMessage =
  | { type: "load" }
  | { type: "classify"; id: number; audio: Float32Array; sampleRate: number };

export type Score = { label: string; score: number };

export type OutMessage =
  | { type: "progress"; status: string; file?: string; progress?: number }
  | { type: "ready" }
  | { type: "result"; id: number; scores: Score[]; ms: number }
  | { type: "error"; id?: number; message: string };

type Options = {
  model: string;
  /** A Hub commit to pin. Browsers cache model files by URL, so a new model must get a new
   * revision or returning visitors keep the old one. */
  revision?: string;
  /** Serve the model from this path/URL prefix instead of the Hugging Face Hub. */
  localModelPath?: string;
  post: (message: OutMessage) => void;
};

/**
 * The only code that knows about the model. Speaks the worker's message protocol so
 * it can be tested in Node without a real Worker.
 */
export function createInferenceHandler({ model, revision, localModelPath, post }: Options) {
  if (localModelPath) {
    env.allowRemoteModels = false;
    env.allowLocalModels = true; // off by default in browsers
    env.localModelPath = localModelPath;
  }
  let classifier: Promise<AudioClassificationPipeline> | undefined;
  let labels: string[] = [];

  const load = () =>
    (classifier ??= pipeline("audio-classification", model, {
      // int8 on WASM: WebGPU has no int8 matmul, and the fp32 weights are too big to ship.
      dtype: "q8",
      revision,
      progress_callback: (p: { status: string; file?: string; progress?: number }) =>
        post({ type: "progress", status: p.status, file: p.file, progress: p.progress }),
    }).then((c) => {
      const { id2label } = c.model.config as unknown as { id2label: Record<string, string> };
      labels = Object.keys(id2label).sort((a, b) => +a - +b).map((i) => id2label[i]);
      return c;
    }));

  return async (msg: InMessage) => {
    const id = msg.type === "classify" ? msg.id : undefined;
    try {
      if (msg.type === "load") {
        await load();
        post({ type: "ready" });
        return;
      }
      if (msg.sampleRate !== SAMPLE_RATE) throw new Error(`expected ${SAMPLE_RATE} Hz audio, got ${msg.sampleRate}`);
      if (msg.audio.length < MIN_SAMPLES) throw new Error(`audio too short: ${msg.audio.length} samples`);
      const c = await load();
      const t0 = performance.now();
      const out = (await c(msg.audio, { top_k: labels.length })) as Score[];
      const byLabel = new Map(out.map((s) => [s.label, s.score]));
      post({ type: "result", id: msg.id, scores: labels.map((label) => ({ label, score: byLabel.get(label) ?? 0 })), ms: performance.now() - t0 });
    } catch (e) {
      post({ type: "error", id, message: e instanceof Error ? e.message : String(e) });
    }
  };
}
