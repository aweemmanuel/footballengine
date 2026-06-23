from __future__ import annotations

import argparse
import json

from .data import generate_sample_matches, load_matches, save_matches
from .ensemble import EnsembleVoter
from .features import FeatureBuilder
from .models import build_default_engines
from .validation import expanding_window_validation, full_walk_forward_report, summarize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Football prediction engine MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Generate sample match data")
    sample.add_argument("--rows", type=int, default=240)
    sample.add_argument("--output", default="outputs/sample_matches.csv")

    validate = sub.add_parser("validate", help="Run expanding-window validation")
    validate.add_argument("--input", default="outputs/sample_matches.csv")
    validate.add_argument("--full-report", action="store_true", help="Include calibration, market benchmark, CLV, and bankroll diagnostics")

    predict = sub.add_parser("predict", help="Predict a future match")
    predict.add_argument("--input", default="outputs/sample_matches.csv")
    predict.add_argument("--home", required=True)
    predict.add_argument("--away", required=True)

    args = parser.parse_args()
    if args.command == "sample":
        df = generate_sample_matches(args.rows)
        save_matches(df, args.output)
        print(f"Wrote {len(df)} matches to {args.output}")
        return

    df = load_matches(args.input)
    builder = FeatureBuilder()
    features, y = builder.transform(df)

    if args.command == "validate":
        if args.full_report:
            print(json.dumps(full_walk_forward_report(features, y), indent=2))
            return
        scores = expanding_window_validation(features, y)
        print(json.dumps({"folds": [score.__dict__ for score in scores], "summary": summarize_scores(scores)}, indent=2))
        return

    if args.command == "predict":
        voter = EnsembleVoter(build_default_engines()).fit(features, y)
        match_features = builder.build_future_match(df, args.home, args.away)
        print(json.dumps(voter.format_match_output(match_features), indent=2))


if __name__ == "__main__":
    main()
