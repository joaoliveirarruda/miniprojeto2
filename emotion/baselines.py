"""Hand-crafted-feature models from the original project, re-evaluated honestly.

- Logistic regression on the 162 features: the floor any model must beat.
- The original Conv1D CNN (ported from Keras to PyTorch, same layers), trained twice:
  with the original protocol (augment everything, then a random split: leaky) and with
  the speaker-independent split (augment train only).

Usage: uv run python -m emotion.baselines
"""


import librosa
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from emotion.data import ARTIFACTS_DIR, RESULTS_DIR, load_index
from emotion.metrics import evaluate, save_results

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# --- the original feature pipeline, kept as-is (22050 Hz, 2.5 s from 0.6 s) ---

def noise(y, rng):
    return y + 0.005 * rng.uniform() * np.amax(y) * rng.normal(size=y.shape[0])


def stretch_pitch(y, sr):
    return librosa.effects.pitch_shift(y=librosa.effects.time_stretch(y=y, rate=0.8), sr=sr, n_steps=0.7)


def features(y, sr):
    return np.hstack([
        np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0),
        np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0),
        np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0),
        np.mean(librosa.feature.rms(y=y).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0),
    ])


def clip_features(path, seed):
    """Features for [clean, noisy, stretched+pitched] versions of one clip: shape (3, 162)."""
    rng = np.random.default_rng(seed)
    y, sr = librosa.load(path, duration=2.5, offset=0.6)
    return np.stack([features(y, sr), features(noise(y, rng), sr), features(stretch_pitch(y, sr), sr)])


def load_features(df):
    cache = ARTIFACTS_DIR / "features.npy"
    if cache.exists():
        return np.load(cache)
    feats = np.stack(Parallel(n_jobs=-1)(delayed(clip_features)(p, i) for i, p in enumerate(df.path)))
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, feats)
    return feats


# --- the original CNN ---

def cnn(n_features, n_classes):
    def block(c_in, c_out):
        return [nn.Conv1d(c_in, c_out, 5, padding=2), nn.ReLU(), nn.MaxPool1d(5, stride=2, padding=2)]

    n = n_features
    for _ in range(4):
        n = (n - 1) // 2 + 1
    return nn.Sequential(
        *block(1, 256), *block(256, 256), *block(256, 128), nn.Dropout(0.2), *block(128, 64),
        nn.Flatten(), nn.Linear(64 * n, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, n_classes),
    )


def train_cnn(x_train, y_train, epochs=50, batch_size=64):
    torch.manual_seed(SEED)
    model = cnn(x_train.shape[1], 8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=2, min_lr=1e-7)
    x = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)[:, None]
    y = torch.tensor(y_train, device=DEVICE)
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(x), device=DEVICE).split(batch_size):
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step(total / len(x))
    model.eval()
    return lambda xs: torch.softmax(model(torch.tensor(xs, dtype=torch.float32, device=DEVICE)[:, None]), -1).detach().cpu().numpy()


def scaled(train, *others):
    scaler = StandardScaler().fit(train)
    return [scaler.transform(a) for a in (train, *others)]


def main():
    df = load_index()
    feats = load_features(df)  # (clips, 3 versions, 162)
    labels = df.label.to_numpy()
    results = {}

    # Original protocol: every augmented copy is a sample, then a random 80/20 split.
    x_all, y_all = feats.reshape(-1, feats.shape[-1]), np.repeat(labels, 3)
    x_tr, x_te, y_tr, y_te = train_test_split(x_all, y_all, test_size=0.2, random_state=SEED)
    x_tr, x_te = scaled(x_tr, x_te)
    results["cnn_leaky"] = evaluate(y_te, train_cnn(x_tr, y_tr)(x_te))

    # Honest protocol: split by actor, augment train only, test on clean clips.
    train, test = (df.split != "test").to_numpy(), (df.split == "test").to_numpy()
    x_tr, y_tr = feats[train].reshape(-1, feats.shape[-1]), np.repeat(labels[train], 3)
    x_te, y_te = feats[test, 0], labels[test]
    x_tr, x_te = scaled(x_tr, x_te)

    logreg = LogisticRegression(max_iter=5000, C=0.1).fit(x_tr, y_tr)
    results["logreg"] = evaluate(y_te, logreg.predict_proba(x_te))
    results["cnn"] = evaluate(y_te, train_cnn(x_tr, y_tr)(x_te))

    save_results(results)
    for name, r in results.items():
        print(f"{name:10s} acc={r['accuracy']:.3f} macro_f1={r['macro_f1']:.3f}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    main()
