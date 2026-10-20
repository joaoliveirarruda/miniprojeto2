"""Export the fine-tuned model to ONNX (fp32 + int8), check parity, measure size/latency.

Writes artifacts/onnx in the layout transformers.js expects (config.json,
preprocessor_config.json, onnx/model.onnx, onnx/model_quantized.onnx), ready to be
pushed to the Hugging Face Hub as-is.

Usage: uv run python -m emotion.export
"""

import json
import shutil
import time

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from emotion.data import ARTIFACTS_DIR, RESULTS_DIR, SAMPLE_RATE, load_audio, load_index
from emotion.predict import MODEL_DIR, Predictor

SRC = ARTIFACTS_DIR / "wav2vec2"


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        return self.model(input_values).logits


def export():
    model = Wav2Vec2ForSequenceClassification.from_pretrained(SRC).eval()
    onnx_dir = MODEL_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    # The TorchScript exporter: the dynamo one writes shape annotations that break the
    # quantizer's shape inference for this model.
    torch.onnx.export(
        Logits(model), (torch.randn(1, 3 * SAMPLE_RATE),), onnx_dir / "model.onnx",
        input_names=["input_values"], output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # Only the transformer matmuls: that's where the weights are, and int8 convs are
    # poorly supported by onnxruntime-web.
    quantize_dynamic(onnx_dir / "model.onnx", onnx_dir / "model_quantized.onnx",
                     weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"])
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(SRC / name, MODEL_DIR / name)


def check_parity():
    """The int8 model must agree with PyTorch on unseen speakers, or the browser isn't
    running what we evaluated."""
    model = Wav2Vec2ForSequenceClassification.from_pretrained(SRC).eval()
    extractor = AutoFeatureExtractor.from_pretrained(SRC)
    q8 = Predictor(quantized=True)
    test = load_index().query("split == 'test'").sample(60, random_state=0)
    diffs, agree = [], []
    for path in test.path:
        y = load_audio(path)
        with torch.no_grad():
            ref = torch.softmax(model(extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values).logits, -1)[0].numpy()
        got = np.array(list(q8.predict(y, SAMPLE_RATE).values()))
        diffs.append(np.abs(ref - got).max())
        agree.append(ref.argmax() == got.argmax())
    return {"max_prob_diff": float(np.max(diffs)), "mean_prob_diff": float(np.mean(diffs)), "argmax_agreement": float(np.mean(agree))}


def measure():
    window = np.random.default_rng(0).normal(scale=0.1, size=3 * SAMPLE_RATE).astype(np.float32)
    q8 = Predictor(quantized=True)
    q8.session.run(None, {"input_values": window[None]})  # warm-up
    t0 = time.perf_counter()
    for _ in range(10):
        q8.predict(window, SAMPLE_RATE)
    onnx_dir = MODEL_DIR / "onnx"
    return {
        "fp32_mb": (onnx_dir / "model.onnx").stat().st_size / 1e6,
        "int8_mb": (onnx_dir / "model_quantized.onnx").stat().st_size / 1e6,
        "int8_ms_per_3s_window_native_cpu": (time.perf_counter() - t0) / 10 * 1000,
    }


if __name__ == "__main__":
    export()
    report = check_parity() | measure()
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert report["argmax_agreement"] >= 0.9, "quantized model disagrees with PyTorch"
