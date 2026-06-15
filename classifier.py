"""Random Forest match classifier.

Decides whether a candidate (Polymarket, Kalshi) pair that cleared the two-stage
matcher is genuinely the *same* event. Random Forest is the active model; the
feature vector is intentionally small so the model trains well on the modest
number of auto-labels accumulated from resolved markets.

Before any labels exist the classifier cold-starts: it accepts every pair that
already passed the Jaccard + cosine gate, so the pipeline is useful on day one
and grows more selective as `labels.csv` fills in.
"""

from __future__ import annotations

import csv
import os
import pickle
import re
import sys
from typing import TYPE_CHECKING

from matching import normalize

if TYPE_CHECKING:
    from matching import MatchPair

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "20"))
LABELS_CSV = os.getenv("LABELS_CSV", "labels.csv")

FEATURE_NAMES = ["jaccard", "cosine", "len_ratio", "num_overlap"]
_NUM_RE = re.compile(r"\d+")


def featurize(pair: "MatchPair") -> list[float]:
    q1, q2 = pair.poly.question, pair.kalshi.question
    len_ratio = min(len(q1), len(q2)) / max(len(q1), len(q2), 1)
    nums1 = set(_NUM_RE.findall(normalize(q1)))
    nums2 = set(_NUM_RE.findall(normalize(q2)))
    if nums1 or nums2:
        num_overlap = len(nums1 & nums2) / len(nums1 | nums2)
    else:
        num_overlap = 1.0  # neither side cites a number -> not a mismatch signal
    return [pair.jaccard, pair.cosine, len_ratio, num_overlap]


class MatchClassifier:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = None
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"[classifier] could not load model: {e}", file=sys.stderr)
                self.model = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, pair: "MatchPair") -> float:
        """Probability that the pair is a true match.

        Cold start (no model) returns 1.0 so the two-stage gate alone drives the
        *alerting* pipeline on day one. A prediction failure returns 0.0
        (reject) — fail-safe, so a broken model never silently accepts matches.
        Live execution additionally requires a trained model (see execution.py).
        """
        if self.model is None:
            return 1.0  # cold start: trust the two-stage gate that already passed
        try:
            proba = self.model.predict_proba([featurize(pair)])[0]
            return float(proba[1])
        except Exception as e:
            print(f"[classifier] predict failed: {e}", file=sys.stderr)
            return 0.0

    def train(self, labels_csv: str = LABELS_CSV) -> bool:
        """Fit a Random Forest from accumulated auto-labels. Returns True on success."""
        if not os.path.exists(labels_csv):
            return False
        X: list[list[float]] = []
        y: list[int] = []
        with open(labels_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    X.append([float(row[name]) for name in FEATURE_NAMES])
                    y.append(int(row["label"]))
                except (KeyError, ValueError):
                    continue
        if len(X) < MIN_TRAIN_ROWS or len(set(y)) < 2:
            return False  # need enough rows and both classes present

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        self.model = model
        return True
