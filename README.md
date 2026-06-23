# Football Prediction Engine

An MVP implementation of a 31-engine ensemble football prediction system. It is built to mirror the attached blueprint while staying runnable without paid API keys or heavyweight ML packages.

## What is included

- Leakage-aware feature engineering with rolling form, rolling xG, Elo, H2H, weather, referee, rest, and odds features.
- A 31-engine registry matching the requested families.
- Independent lightweight engines with different feature subsets, weights, and seeds.
- Hard majority voting and weighted probability aggregation.
- Rolling-origin season validation with market benchmarks, calibration metrics, CLV-style edge checks, and bankroll simulation.
- Training-only probability calibration using temperature scaling plus a learned closing-market blend.
- Tipster-market correlation diagnostics to avoid double-counting consensus signals that are just shaded bookmaker odds.
- Uncertainty intervals and directional disagreement flags in prediction output.
- JSON prediction output in the requested shape.
- Synthetic sample data so the project works immediately.

## Quick start

```powershell
python -m football_prediction_engine.cli sample --rows 240 --output outputs/sample_matches.csv
python -m football_prediction_engine.cli validate --input outputs/sample_matches.csv
python -m football_prediction_engine.cli validate --input outputs/sample_matches.csv --full-report
python -m football_prediction_engine.cli predict --input outputs/sample_matches.csv --home Arsenal --away Manchester City
```

## API deployment

The project includes a FastAPI wrapper for hosted use:

```powershell
pip install -r requirements.txt
python -m football_prediction_engine.cli sample --rows 240 --output outputs/sample_matches.csv
uvicorn football_prediction_engine.api:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs`.

For Railway deployment, push this folder to GitHub and deploy it as a Railway service. The included `railway.toml`, `runtime.txt`, and `Procfile` start the API automatically. Full steps are in [docs/deploy_and_use.md](C:/Users/England/Documents/Codex/2026-06-23/hi/docs/deploy_and_use.md).

## Project layout

```text
football_prediction_engine/
  cli.py          command line entry points
  api.py          FastAPI service for hosted predictions
  data.py         sample data and CSV loading
  ensemble.py     voting, confidence, JSON formatting
  evaluation.py   calibration, market benchmark, bankroll backtest
  features.py     leakage-aware feature pipeline
  models.py       31-engine registry and runnable engines
  validation.py   expanding-window evaluation
tests/
  test_engine.py
```

## Live data adapters

The current implementation is API-ready but offline by default. Add real clients for FootyStats, FBref, OpenWeatherMap, Rotowire, and odds providers behind `football_prediction_engine.data.load_matches`, then keep the downstream feature, model, and ensemble layers unchanged.

## Professional evaluation stance

Closing odds are treated as the benchmark model. The system reports whether it beats that benchmark on accuracy, Brier score, log loss, and simulated value betting. See [docs/professional_evaluation_protocol.md](C:/Users/England/Documents/Codex/2026-06-23/hi/docs/professional_evaluation_protocol.md) for the walk-forward, calibration, CLV, and bankroll protocol.

## Notes

The bundled engines are deterministic MVP stand-ins for the full production models. They are intentionally dependency-light so they run on a clean machine. The registry names the full target ensemble, and each engine can be swapped for XGBoost, LightGBM, CatBoost, PyTorch, or LLM-backed implementations as those dependencies and credentials become available.
