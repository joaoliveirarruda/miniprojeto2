import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from emotion.data import EMOTIONS, RESULTS_DIR

RESULTS_FILE = RESULTS_DIR / "metrics.json"


def evaluate(y_true, probs) -> dict:
    y_pred = np.asarray(probs).argmax(-1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class_f1": dict(zip(EMOTIONS, f1_score(y_true, y_pred, average=None, labels=range(8)).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(8)).tolist(),
        "n": len(y_true),
    }


def save_results(results: dict):
    """Merge `results` into the shared metrics file, so each script owns its own keys."""
    RESULTS_DIR.mkdir(exist_ok=True)
    merged = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}
    RESULTS_FILE.write_text(json.dumps(merged | results, indent=2))
