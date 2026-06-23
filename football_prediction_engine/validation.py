from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .calibration import ProbabilityCalibrator
from .ensemble import EnsembleVoter
from .evaluation import evaluation_report, market_probabilities
from .models import build_default_engines


@dataclass
class FoldScore:
    train_start: int
    train_end: int
    validate_year: int
    accuracy: float
    market_accuracy: float
    brier_score: float
    market_brier_score: float
    log_loss: float
    market_log_loss: float
    calibrator_temperature: float
    calibrator_market_blend: float
    roi: float
    max_drawdown: float
    value_candidates: int
    samples: int


def expanding_window_validation(features: pd.DataFrame, y: pd.Series, min_train_years: int = 3) -> list[FoldScore]:
    years = sorted(int(year) for year in features["season"].unique())
    scores: list[FoldScore] = []
    for index in range(min_train_years, len(years)):
        train_years = years[:index]
        validate_year = years[index]
        train_mask = features["season"].isin(train_years)
        val_mask = features["season"] == validate_year
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        voter = EnsembleVoter(build_default_engines())
        voter.fit(features.loc[train_mask], y.loc[train_mask])
        val_features = features.loc[val_mask]
        train_features = features.loc[train_mask]
        calibrator = ProbabilityCalibrator().fit(
            voter.predict_proba(train_features),
            market_probabilities(train_features),
            y.loc[train_mask],
        )
        probabilities = calibrator.predict(voter.predict_proba(val_features), market_probabilities(val_features))
        report = evaluation_report(y.loc[val_mask], probabilities, val_features)
        scores.append(
            FoldScore(
                min(train_years),
                max(train_years),
                validate_year,
                float(report["accuracy"]),
                float(report["market_accuracy"]),
                float(report["brier_score"]),
                float(report["market_brier_score"]),
                float(report["log_loss"]),
                float(report["market_log_loss"]),
                calibrator.temperature_,
                calibrator.market_blend_,
                float(report["bankroll"]["roi"]),
                float(report["bankroll"]["max_drawdown"]),
                int(report["value_candidates"]),
                int(val_mask.sum()),
            )
        )
    return scores


def summarize_scores(scores: list[FoldScore]) -> dict[str, float | int]:
    if not scores:
        return {"folds": 0, "mean_accuracy": 0.0, "std_accuracy": 0.0}
    values = np.asarray([score.accuracy for score in scores], dtype=float)
    market_values = np.asarray([score.market_accuracy for score in scores], dtype=float)
    brier = np.asarray([score.brier_score for score in scores], dtype=float)
    market_brier = np.asarray([score.market_brier_score for score in scores], dtype=float)
    roi = np.asarray([score.roi for score in scores], dtype=float)
    blends = np.asarray([score.calibrator_market_blend for score in scores], dtype=float)
    return {
        "folds": len(scores),
        "mean_accuracy": round(float(values.mean()), 4),
        "std_accuracy": round(float(values.std()), 4),
        "mean_market_accuracy": round(float(market_values.mean()), 4),
        "mean_brier_score": round(float(brier.mean()), 4),
        "mean_market_brier_score": round(float(market_brier.mean()), 4),
        "mean_roi": round(float(roi.mean()), 4),
        "mean_calibrator_market_blend": round(float(blends.mean()), 4),
        "total_value_candidates": int(sum(score.value_candidates for score in scores)),
        "total_validation_samples": int(sum(score.samples for score in scores)),
    }


def full_walk_forward_report(features: pd.DataFrame, y: pd.Series, min_train_years: int = 3) -> dict[str, Any]:
    scores = expanding_window_validation(features, y, min_train_years)
    voter = EnsembleVoter(build_default_engines()).fit(features, y)
    calibrator = ProbabilityCalibrator().fit(voter.predict_proba(features), market_probabilities(features), y)
    probabilities = calibrator.predict(voter.predict_proba(features), market_probabilities(features))
    return {
        "protocol": "rolling-origin walk-forward by season; each fold trains only on seasons before the validation season",
        "folds": [score.__dict__ for score in scores],
        "summary": summarize_scores(scores),
        "calibration_layer": {
            "method": "temperature scaling plus learned market blend",
            "fit_scope": "training seasons only inside each fold",
            "full_sample_temperature": round(calibrator.temperature_, 4),
            "full_sample_market_blend": round(calibrator.market_blend_, 4),
        },
        "full_sample_diagnostics": evaluation_report(y, probabilities, features),
    }
