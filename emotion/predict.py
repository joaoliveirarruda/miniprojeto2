"""Inference with the exported ONNX model: the same file the browser runs."""

import json
import os
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort

from emotion.data import ARTIFACTS_DIR, SAMPLE_RATE

MODEL_DIR = Path(os.environ.get("EMOTION_MODEL_DIR", ARTIFACTS_DIR / "onnx"))
MIN_SAMPLES = 400  # wav2vec2's conv front-end needs at least one 25 ms frame


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR, quantized: bool = True):
        name = "model_quantized.onnx" if quantized else "model.onnx"
        self.session = ort.InferenceSession(str(model_dir / "onnx" / name), providers=["CPUExecutionProvider"])
        config = json.loads((model_dir / "config.json").read_text())
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        self.normalize = json.loads((model_dir / "preprocessor_config.json").read_text())["do_normalize"]

    def predict(self, waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
        """Mono waveform at any sample rate -> {emotion: probability}, in label order."""
        y = np.asarray(waveform, dtype=np.float32)
        if sample_rate != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        y = np.pad(y, (0, max(0, MIN_SAMPLES - len(y))))
        if self.normalize:
            y = (y - y.mean()) / np.sqrt(y.var() + 1e-7)
        (logits,) = self.session.run(None, {"input_values": y[None].astype(np.float32)})
        probs = np.exp(logits[0] - logits[0].max())
        probs /= probs.sum()
        return dict(zip(self.labels, probs.tolist()))
