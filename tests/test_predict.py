import librosa
import numpy as np
import pytest
import soundfile as sf

from emotion.data import EMOTIONS, ROOT
from emotion.predict import MODEL_DIR, Predictor

FIXTURES = ROOT / "fixtures"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "onnx" / "model_quantized.onnx").exists(),
    reason="exported model missing: run `python -m emotion.export` (or download it from the Hub)",
)


@pytest.fixture(scope="module")
def predictor():
    return Predictor()


def fixture(name):
    y, sr = sf.read(FIXTURES / f"{name}.wav", dtype="float32")
    return y, sr


def test_returns_a_distribution_over_all_emotions_in_label_order(predictor):
    probs = predictor.predict(*fixture("angry"))
    assert list(probs) == EMOTIONS
    assert sum(probs.values()) == pytest.approx(1, abs=1e-4)
    assert all(0 <= p <= 1 for p in probs.values())


@pytest.mark.parametrize("seconds", [0.3, 3, 10])
def test_accepts_any_duration(predictor, seconds):
    y = np.random.default_rng(0).normal(scale=0.1, size=int(16_000 * seconds)).astype(np.float32)
    assert len(predictor.predict(y, 16_000)) == len(EMOTIONS)


def test_resamples_other_rates_to_the_same_answer(predictor):
    y, sr = fixture("sad")
    at_16k = predictor.predict(y, sr)
    at_44k = predictor.predict(librosa.resample(y, orig_sr=sr, target_sr=44_100), 44_100)
    assert max(at_16k, key=at_16k.get) == max(at_44k, key=at_44k.get)
    assert np.allclose(list(at_16k.values()), list(at_44k.values()), atol=0.05)


def test_silence_does_not_crash(predictor):
    assert len(predictor.predict(np.zeros(16_000, dtype=np.float32), 16_000)) == len(EMOTIONS)


def test_recognises_most_unseen_speaker_fixtures(predictor):
    # Clips from test actors the model never trained on. A quality floor, not a target:
    # it fails if export/quantization or preprocessing silently breaks the model.
    hits = [max(p := predictor.predict(*fixture(e)), key=p.get) == e for e in EMOTIONS]
    assert sum(hits) >= 5, dict(zip(EMOTIONS, hits))
