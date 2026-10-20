import { createInferenceHandler, type InMessage } from "./inference";

const local = process.env.NEXT_PUBLIC_MODEL_LOCAL === "1";

const handle = createInferenceHandler({
  model: local ? "speech-emotion" : (process.env.NEXT_PUBLIC_MODEL_ID ?? "joaoliveirarruda/speech-emotion-wav2vec2"),
  revision: process.env.NEXT_PUBLIC_MODEL_REVISION,
  localModelPath: local ? `${self.location.origin}/models/` : undefined,
  post: (m) => self.postMessage(m),
});

self.onmessage = (e: MessageEvent<InMessage>) => handle(e.data);
