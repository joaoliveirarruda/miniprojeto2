"""Fine-tune a pretrained wav2vec2 on RAVDESS with the speaker-independent split.

The model reads raw 16 kHz audio, so there is no hand-made feature pipeline that could
drift between training and the web app. The best epoch (val macro-F1) is saved to
artifacts/wav2vec2 and scored on the test actors.

Usage: uv run python -m emotion.finetune [--layers 12] [--epochs 20]
"""

import argparse
import copy

import numpy as np
import torch
from joblib import Parallel, delayed
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

from emotion.data import ARTIFACTS_DIR, EMOTIONS, SAMPLE_RATE, load_audio, load_index
from emotion.metrics import evaluate, save_results

BASE_MODEL = "facebook/wav2vec2-base"
WINDOW = 3 * SAMPLE_RATE  # the web app classifies 3 s windows
SEED = 42


def random_window(y, rng):
    """A WINDOW-long random crop of `y`, zero-padded at a random offset if shorter."""
    if len(y) >= WINDOW:
        start = rng.integers(0, len(y) - WINDOW + 1)
        return y[start:start + WINDOW]
    out = np.zeros(WINDOW, dtype=np.float32)
    start = rng.integers(0, WINDOW - len(y) + 1)
    out[start:start + len(y)] = y
    return out


def add_noise(y, rng):
    """White noise at a random 15-40 dB SNR, half of the time: mics in the wild are noisy."""
    if rng.uniform() < 0.5:
        return y
    snr_db = rng.uniform(15, 40)
    power = np.mean(y**2) / 10 ** (snr_db / 10)
    return (y + rng.normal(scale=np.sqrt(power), size=y.shape)).astype(np.float32)


def predict_probs(model, extractor, clips, device):
    model.eval()
    out = []
    with torch.no_grad():
        for y in clips:
            x = extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
            out.append(torch.softmax(model(x).logits, -1)[0].cpu().numpy())
    return np.stack(out)


def main(layers, epochs, batch_size, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = load_index()
    audio = Parallel(n_jobs=-1)(delayed(load_audio)(p) for p in df.path)
    split = {s: [i for i in range(len(df)) if df.split[i] == s] for s in ("train", "val", "test")}
    labels = df.label.to_numpy()

    extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(EMOTIONS),
        id2label=dict(enumerate(EMOTIONS)),
        label2id={e: i for i, e in enumerate(EMOTIONS)},
        num_hidden_layers=layers,
    ).to(device)
    model.freeze_feature_encoder()

    head = [p for n, p in model.named_parameters() if n.startswith(("projector", "classifier"))]
    body = [p for n, p in model.named_parameters() if not n.startswith(("projector", "classifier")) and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 1e-3}])
    steps = epochs * -(-len(split["train"]) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-5, 1e-3], total_steps=steps, pct_start=0.1)

    best_f1, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(split["train"])
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            batch = [add_noise(random_window(audio[i], rng), rng) for i in idx]
            x = extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
            loss = model(x, labels=torch.tensor(labels[idx], device=device)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = evaluate(labels[split["val"]], predict_probs(model, extractor, [audio[i] for i in split["val"]], device))
        print(f"epoch {epoch + 1:2d} loss={np.mean(losses):.3f} val_acc={val['accuracy']:.3f} val_f1={val['macro_f1']:.3f}", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    out_dir = ARTIFACTS_DIR / "wav2vec2"
    model.save_pretrained(out_dir)
    extractor.save_pretrained(out_dir)

    test = evaluate(labels[split["test"]], predict_probs(model, extractor, [audio[i] for i in split["test"]], device))
    test["val_macro_f1"] = best_f1
    test["layers"] = layers
    save_results({"wav2vec2": test})
    print(f"test acc={test['accuracy']:.3f} macro_f1={test['macro_f1']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    main(**vars(parser.parse_args()))
