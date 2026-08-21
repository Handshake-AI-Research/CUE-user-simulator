"""RandomForest human-likeness discriminator + Chamfer coverage (PPol)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

DISCRIMINATOR_FILE = "discriminator.pkl"


class HumanLikenessDiscriminator:
    """Distinguishes real user fingerprints from synthetic ones.

    ``score`` returns P(real); higher means more human-like. Falls back to a
    distance-to-real heuristic when scikit-learn is unavailable or unfit.
    """

    def __init__(self) -> None:
        self.model = None
        self._real_mean: np.ndarray | None = None
        self._real_std: np.ndarray | None = None

    def fit(self, real_x: np.ndarray, synth_x: np.ndarray) -> "HumanLikenessDiscriminator":
        real_x = np.atleast_2d(real_x).astype(float)
        self._real_mean = real_x.mean(axis=0)
        self._real_std = real_x.std(axis=0) + 1e-6
        try:
            from sklearn.ensemble import RandomForestClassifier

            if len(synth_x) == 0:
                raise ValueError("no synthetic negatives")
            synth_x = np.atleast_2d(synth_x).astype(float)
            x = np.vstack([real_x, synth_x])
            y = np.concatenate([np.ones(len(real_x)), np.zeros(len(synth_x))])
            model = RandomForestClassifier(n_estimators=200, random_state=0)
            model.fit(x, y)
            self.model = model
        except Exception as exc:  # noqa: BLE001
            print(f"[ppol] RandomForest unavailable ({exc}); using distance heuristic.")
            self.model = None
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(float)
        if self.model is not None:
            classes = list(self.model.classes_)
            idx = classes.index(1.0) if 1.0 in classes else 1
            return self.model.predict_proba(x)[:, idx]
        if self._real_mean is None:
            return np.full(len(x), 0.5)
        z = (x - self._real_mean) / self._real_std
        dist = np.linalg.norm(z, axis=1)
        return 1.0 / (1.0 + dist / np.sqrt(x.shape[1]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "model": self.model,
                    "real_mean": self._real_mean,
                    "real_std": self._real_std,
                },
                handle,
            )

    @classmethod
    def load(cls, path: Path) -> "HumanLikenessDiscriminator":
        obj = cls()
        if path.exists():
            with open(path, "rb") as handle:
                state = pickle.load(handle)
            obj.model = state.get("model")
            obj._real_mean = state.get("real_mean")
            obj._real_std = state.get("real_std")
        return obj


def chamfer_distance(synth_x: np.ndarray, real_x: np.ndarray) -> float:
    """Symmetric Chamfer distance between two fingerprint point sets (lower=better)."""

    synth_x = np.atleast_2d(synth_x).astype(float)
    real_x = np.atleast_2d(real_x).astype(float)
    if len(synth_x) == 0 or len(real_x) == 0:
        return float("inf")
    std = real_x.std(axis=0) + 1e-6
    s = synth_x / std
    r = real_x / std
    d = np.linalg.norm(s[:, None, :] - r[None, :, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())
