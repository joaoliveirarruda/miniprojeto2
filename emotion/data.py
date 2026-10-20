"""RAVDESS speech: download, metadata index and the speaker-independent split."""

import io
import urllib.request
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

# Alphabetical, matching the original project's one-hot order. This is the single
# source of truth for label ids; the exported model's config carries it to the web app.
EMOTIONS = ["angry", "calm", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
_CODE_TO_EMOTION = {
    1: "neutral", 2: "calm", 3: "happy", 4: "sad",
    5: "angry", 6: "fear", 7: "disgust", 8: "surprise",
}

# Fixed actor split, 50/50 male (odd ids) and female (even ids) in val and test.
VAL_ACTORS = {15, 16, 17, 18}
TEST_ACTORS = {19, 20, 21, 22, 23, 24}

SAMPLE_RATE = 16_000


def download():
    if any(RAW_DIR.glob("Actor_*")):
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(RAW_DIR)


def parse_name(name: str) -> dict:
    """`03-01-05-02-01-01-12.wav` -> metadata. See the RAVDESS docs for the field order."""
    _, channel, emotion, intensity, statement, repetition, actor = map(int, Path(name).stem.split("-"))
    assert channel == 1, f"{name} is not speech"
    return {
        "emotion": _CODE_TO_EMOTION[emotion],
        "intensity": "strong" if intensity == 2 else "normal",
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "gender": "male" if actor % 2 else "female",
    }


def split_of(actor: int) -> str:
    return "test" if actor in TEST_ACTORS else "val" if actor in VAL_ACTORS else "train"


def load_index() -> pd.DataFrame:
    download()
    rows = [{"path": str(p), **parse_name(p.name)} for p in sorted(RAW_DIR.glob("Actor_*/*.wav"))]
    df = pd.DataFrame(rows)
    df["split"] = df["actor"].map(split_of)
    df["label"] = df["emotion"].map(EMOTIONS.index)
    return df


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono audio at `sr`, with the leading/trailing silence of the recordings trimmed."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return librosa.effects.trim(y, top_db=30)[0]


if __name__ == "__main__":
    df = load_index()
    print(df.groupby("split").agg(clips=("path", "size"), actors=("actor", "nunique")))
    assert len(df) == 1440
    assert not set(df[df.split == "test"].actor) & set(df[df.split == "train"].actor)
