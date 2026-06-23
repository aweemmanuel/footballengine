import unittest

from football_prediction_engine.data import generate_sample_matches
from football_prediction_engine.ensemble import EnsembleVoter
from football_prediction_engine.features import FeatureBuilder
from football_prediction_engine.models import ENGINE_SPECS, build_default_engines
from football_prediction_engine.validation import expanding_window_validation, full_walk_forward_report, summarize_scores


class FootballPredictionEngineTest(unittest.TestCase):
    def test_feature_builder_creates_leakage_safe_rows(self):
        matches = generate_sample_matches(rows=80, seed=3)
        features, y = FeatureBuilder().transform(matches)
        self.assertEqual(len(features), len(matches))
        self.assertEqual(len(y), len(matches))
        self.assertIn("elo_diff", features.columns)
        self.assertIn("xg_diff_5", features.columns)
        self.assertIn("form_diff_5", features.columns)

    def test_default_registry_has_31_engines(self):
        self.assertEqual(len(ENGINE_SPECS), 31)
        self.assertEqual(len(build_default_engines()), 31)

    def test_ensemble_prediction_output_shape(self):
        matches = generate_sample_matches(rows=120, seed=4)
        builder = FeatureBuilder()
        features, y = builder.transform(matches)
        voter = EnsembleVoter(build_default_engines()).fit(features.iloc[:90], y.iloc[:90])
        output = voter.format_match_output(features.iloc[[91]].reset_index(drop=True))
        self.assertIn(output["most_likely"], {"home_win", "draw", "away_win"})
        self.assertEqual(sum(output["engine_breakdown"].values()), 31)
        self.assertEqual(set(output["predictions"]), {"home_win", "draw", "away_win"})
        self.assertIn("uncertainty", output)
        self.assertIn("market_edge", output)

    def test_expanding_window_validation_runs(self):
        matches = generate_sample_matches(rows=160, seed=5)
        features, y = FeatureBuilder().transform(matches)
        scores = expanding_window_validation(features, y, min_train_years=2)
        summary = summarize_scores(scores)
        self.assertGreater(summary["folds"], 0)
        self.assertGreaterEqual(summary["mean_accuracy"], 0.0)
        self.assertLessEqual(summary["mean_accuracy"], 1.0)
        self.assertIn("mean_market_accuracy", summary)
        self.assertIn("mean_brier_score", summary)
        self.assertIn("mean_calibrator_market_blend", summary)

    def test_full_report_includes_market_and_tipster_diagnostics(self):
        matches = generate_sample_matches(rows=160, seed=6)
        features, y = FeatureBuilder().transform(matches)
        report = full_walk_forward_report(features, y, min_train_years=2)
        diagnostics = report["full_sample_diagnostics"]
        self.assertIn("market_brier_score", diagnostics)
        self.assertIn("calibration_layer", report)
        self.assertIn("bankroll", diagnostics)
        self.assertIn("tipster_market_correlation", diagnostics)

    def test_api_health(self):
        try:
            from fastapi.testclient import TestClient

            from football_prediction_engine.api import app
        except ModuleNotFoundError:
            self.skipTest("FastAPI is optional for API deployment tests")
        client = TestClient(app)
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
