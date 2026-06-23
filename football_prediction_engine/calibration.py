from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import brier_score


class ProbabilityCalibrator:
    """Temperature scaling plus optional market blend, selected on training data only."""

    def __init__(self) -> None:
        self.temperature_: float = 1.0
        self.market_blend_: float = 0.0

    def fit(self, probabilities: np.ndarray, market_probabilities: np.ndarray, y: pd.Series) -> "ProbabilityCalibrator":
        best_score = float("inf")
        best_temperature = 1.0
        best_blend = 0.0
        for temperature in np.linspace(0.85, 2.75, 9):
            tempered = self._temperature_scale(probabilities, float(temperature))
            for blend in np.linspace(0.0, 0.55, 12):
                calibrated = (1.0 - blend) * tempered + blend * market_probabilities
                score = brier_score(y, calibrated)
                if score < best_score:
                    best_score = score
                    best_temperature = float(temperature)
                    best_blend = float(blend)
        self.temperature_ = best_temperature
        self.market_blend_ = best_blend
        return self

    def predict(self, probabilities: np.ndarray, market_probabilities: np.ndarray) -> np.ndarray:
        tempered = self._temperature_scale(probabilities, self.temperature_)
        calibrated = (1.0 - self.market_blend_) * tempered + self.market_blend_ * market_probabilities
        return calibrated / calibrated.sum(axis=1, keepdims=True)

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        clipped = np.clip(probabilities, 1e-8, 1.0)
        logits = np.log(clipped)
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)
