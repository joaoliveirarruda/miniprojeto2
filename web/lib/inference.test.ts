import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import { EMOTIONS_IN_TEST_ORDER, readWav } from "./test-utils";
import { createInferenceHandler, type OutMessage } from "./inference";

// Local export (python -m emotion.export) by default; CI points MODEL_ID at the Hub.
const LOCAL_DIR = resolve(__dirname, "../../artifacts");
const MODEL_ID = process.env.MODEL_ID ?? "onnx";
const hasModel = process.env.MODEL_ID !== undefined || existsSync(resolve(LOCAL_DIR, "onnx/onnx/model_quantized.onnx"));

const fixture = (name: string) => readWav(readFileSync(resolve(__dirname, `../../fixtures/${name}.wav`)));

describe.skipIf(!hasModel)("inference worker", () => {
  const messages: OutMessage[] = [];
  const handle = createInferenceHandler({
    model: MODEL_ID,
    localModelPath: process.env.MODEL_ID ? undefined : LOCAL_DIR + "/",
    post: (m) => messages.push(m),
  });
  const classify = async (id: number, audio: Float32Array, sampleRate = 16000) => {
    await handle({ type: "classify", id, audio, sampleRate });
    return messages.findLast((m) => "id" in m && m.id === id)!;
  };

  beforeAll(async () => {
    await handle({ type: "load" });
  }, 120_000);

  it("reports loading progress, then ready", () => {
    expect(messages.some((m) => m.type === "progress")).toBe(true);
    expect(messages.findLast((m) => m.type !== "progress")?.type).toBe("ready");
  });

  it("returns probabilities for all 8 emotions, in label order, summing to 1", async () => {
    const { audio, sampleRate } = fixture("angry");
    const result = await classify(1, audio, sampleRate);
    if (result.type !== "result") throw new Error(JSON.stringify(result));
    expect(result.scores.map((s) => s.label)).toEqual(EMOTIONS_IN_TEST_ORDER);
    expect(result.scores.reduce((a, s) => a + s.score, 0)).toBeCloseTo(1, 3);
    expect(result.ms).toBeGreaterThan(0);
  });

  it("puts the fixture's emotion on top", async () => {
    const result = await classify(2, fixture("angry").audio);
    if (result.type !== "result") throw new Error(JSON.stringify(result));
    const top = result.scores.reduce((a, b) => (b.score > a.score ? b : a));
    expect(top.label).toBe("angry");
  });

  it("reports an error (not a crash) for empty audio", async () => {
    expect((await classify(3, new Float32Array(0))).type).toBe("error");
  });

  it("reports an error for audio that is not 16 kHz", async () => {
    expect((await classify(4, fixture("angry").audio, 44100)).type).toBe("error");
  });
});
