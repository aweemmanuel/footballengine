from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import OUTCOMES


@dataclass(frozen=True)
class EngineSpec:
    name: str
    family: str
    features: tuple[str, ...]
    seed: int
    validation_weight: float = 1.0


ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec("xgboost_proxy", "tree", ("elo_diff", "xg_diff_5", "form_diff_5", "odds_home_implied"), 11, 1.04),
    EngineSpec("lightgbm_proxy", "tree", ("elo_diff", "xg_diff_10", "venue_form_diff_5", "odds_away_implied"), 12, 1.03),
    EngineSpec("catboost_proxy", "tree", ("xg_diff_38", "form_diff_10", "rest_days_diff", "match_importance"), 13, 1.02),
    EngineSpec("random_forest_proxy", "tree", ("elo_diff", "form_diff_15", "h2h_goal_diff_5", "wind_speed"), 14, 1.00),
    EngineSpec("extra_trees_proxy", "tree", ("xga_diff_10", "venue_form_diff_5", "temperature", "humidity"), 15, 0.99),
    EngineSpec("gradient_boosting_proxy", "tree", ("elo_diff", "xg_diff_15", "odds_home_movement", "odds_away_movement"), 16, 1.01),
    EngineSpec("logistic_regression_proxy", "linear", ("elo_diff", "xg_diff_5", "form_diff_5"), 21, 1.00),
    EngineSpec("ridge_classifier_proxy", "linear", ("elo_diff", "xg_diff_10", "form_diff_10"), 22, 0.98),
    EngineSpec("svm_linear_proxy", "linear", ("elo_diff", "venue_form_diff_5", "rest_days_diff"), 23, 0.98),
    EngineSpec("svm_rbf_proxy", "linear", ("xg_diff_5", "xga_diff_5", "match_importance"), 24, 0.97),
    EngineSpec("poisson_proxy", "statistical", ("home_xg_for_10", "away_xg_for_10", "home_xga_10", "away_xga_10"), 25, 1.02),
    EngineSpec("dixon_coles_proxy", "statistical", ("home_xg_for_5", "away_xg_for_5", "h2h_draws_5", "referee_penalties_per_match"), 26, 1.01),
    EngineSpec("knn_k3_proxy", "nearest_neighbor", ("elo_diff", "form_diff_5", "xg_diff_5"), 31, 0.96),
    EngineSpec("knn_k5_proxy", "nearest_neighbor", ("elo_diff", "form_diff_10", "xg_diff_10"), 32, 0.96),
    EngineSpec("knn_k7_proxy", "nearest_neighbor", ("elo_diff", "form_diff_15", "xg_diff_15"), 33, 0.95),
    EngineSpec("mlp_3_layer_proxy", "neural", ("elo_diff", "xg_diff_5", "xga_diff_5", "odds_home_implied"), 41, 1.00),
    EngineSpec("mlp_5_layer_proxy", "neural", ("elo_diff", "xg_diff_10", "xga_diff_10", "odds_away_implied"), 42, 1.00),
    EngineSpec("tabnet_proxy", "neural", ("form_diff_5", "venue_form_diff_5", "rest_days_diff", "match_importance"), 43, 0.99),
    EngineSpec("tabular_transformer_proxy", "neural", ("elo_diff", "xg_diff_38", "form_diff_38", "odds_draw_implied"), 44, 0.99),
    EngineSpec("lstm_form_proxy", "neural", ("form_diff_5", "form_diff_10", "form_diff_15", "form_diff_38"), 45, 0.98),
    EngineSpec("gaussian_naive_bayes_proxy", "bayesian", ("elo_diff", "xg_diff_5", "rest_days_diff"), 51, 0.95),
    EngineSpec("bayesian_glm_proxy", "bayesian", ("elo_diff", "xg_diff_10", "form_diff_10", "humidity"), 52, 0.97),
    EngineSpec("bayesian_ridge_proxy", "bayesian", ("elo_diff", "venue_form_diff_5", "xga_diff_15"), 53, 0.97),
    EngineSpec("gpt4_agent_proxy", "llm", ("odds_home_implied", "odds_draw_implied", "odds_away_implied", "match_importance"), 61, 0.94),
    EngineSpec("claude_agent_proxy", "llm", ("elo_diff", "form_diff_5", "referee_cards_per_match", "precipitation_probability"), 62, 0.94),
    EngineSpec("llama_agent_proxy", "llm", ("xg_diff_5", "h2h_home_wins_5", "h2h_away_wins_5", "wind_speed"), 63, 0.93),
    EngineSpec("mixtral_agent_proxy", "llm", ("venue_form_diff_5", "rest_days_diff", "temperature", "humidity"), 64, 0.93),
    EngineSpec("gemini_agent_proxy", "llm", ("elo_diff", "odds_home_movement", "odds_away_movement", "stadium_altitude"), 65, 0.93),
    EngineSpec("stacking_xgb_meta_proxy", "ensemble_meta", ("elo_diff", "xg_diff_5", "form_diff_5", "odds_home_implied"), 71, 1.05),
    EngineSpec("blending_proxy", "ensemble_meta", ("elo_diff", "xg_diff_10", "venue_form_diff_5", "odds_draw_implied"), 72, 1.04),
    EngineSpec("weighted_voting_proxy", "ensemble_meta", ("odds_home_implied", "odds_draw_implied", "odds_away_implied", "elo_diff"), 73, 1.06),
)


class RuleBasedEngine:
    """Small deterministic model that can be replaced by a production learner."""

    classes_ = OUTCOMES

    def __init__(self, spec: EngineSpec):
        self.spec = spec
        self.feature_means_: pd.Series | None = None
        self.feature_stds_: pd.Series | None = None
        self.class_priors_: dict[str, float] = {label: 1 / 3 for label in OUTCOMES}
        rng = np.random.default_rng(spec.seed)
        self.coefficients_ = rng.normal(0.0, 0.35, size=len(spec.features))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleBasedEngine":
        data = self._feature_frame(X)
        self.feature_means_ = data.mean()
        self.feature_stds_ = data.std().replace(0, 1).fillna(1)
        counts = y.value_counts(normalize=True).to_dict()
        self.class_priors_ = {label: float(counts.get(label, 0.001)) for label in OUTCOMES}
        total = sum(self.class_priors_.values())
        self.class_priors_ = {label: value / total for label, value in self.class_priors_.items()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.asarray([OUTCOMES[int(np.argmax(row))] for row in proba])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        data = self._feature_frame(X)
        if self.feature_means_ is None or self.feature_stds_ is None:
            self.feature_means_ = data.mean()
            self.feature_stds_ = data.std().replace(0, 1).fillna(1)
        z = ((data - self.feature_means_) / self.feature_stds_).fillna(0).to_numpy(dtype=float)
        signal = z @ self.coefficients_

        # Anchor known football signals so every proxy has sensible behavior.
        elo = X.get("elo_diff", pd.Series(0, index=X.index)).to_numpy(dtype=float) / 420.0
        xg = X.get("xg_diff_5", X.get("xg_diff_10", pd.Series(0, index=X.index))).to_numpy(dtype=float)
        form = X.get("form_diff_5", X.get("form_diff_10", pd.Series(0, index=X.index))).to_numpy(dtype=float) / 3.0
        odds_home = X.get("odds_home_implied", pd.Series(self.class_priors_["home_win"], index=X.index)).to_numpy(dtype=float)
        odds_draw = X.get("odds_draw_implied", pd.Series(self.class_priors_["draw"], index=X.index)).to_numpy(dtype=float)
        odds_away = X.get("odds_away_implied", pd.Series(self.class_priors_["away_win"], index=X.index)).to_numpy(dtype=float)

        home_score = np.log(self.class_priors_["home_win"]) + 0.9 * elo + 0.55 * xg + 0.35 * form + 0.55 * signal + odds_home
        away_score = np.log(self.class_priors_["away_win"]) - 0.9 * elo - 0.55 * xg - 0.35 * form - 0.55 * signal + odds_away
        draw_score = np.log(self.class_priors_["draw"]) - 0.45 * np.abs(elo + xg) + 0.30 * odds_draw
        scores = np.vstack([home_score, draw_score, away_score]).T
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def _feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: pd.to_numeric(X.get(name, 0.0), errors="coerce").fillna(0.0) for name in self.spec.features}, index=X.index)


def build_default_engines() -> list[RuleBasedEngine]:
    return [RuleBasedEngine(spec) for spec in ENGINE_SPECS]
