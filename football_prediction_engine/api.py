from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import generate_sample_matches, load_matches, save_matches
from .ensemble import EnsembleVoter
from .features import FeatureBuilder
from .models import build_default_engines
from .validation import full_walk_forward_report


class PredictionRequest(BaseModel):
    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Manchester City"])
    match_id: str = "api_match"
    season: int | None = None
    odds_home_implied: float = 0.40
    odds_draw_implied: float = 0.28
    odds_away_implied: float = 0.32


@lru_cache(maxsize=1)
def _engine_context() -> tuple[FeatureBuilder, Any, Any, EnsembleVoter]:
    sample_path = Path("outputs/sample_matches.csv")
    if not sample_path.exists():
        save_matches(generate_sample_matches(rows=240), sample_path)
    matches = load_matches(sample_path)
    builder = FeatureBuilder()
    features, y = builder.transform(matches)
    voter = EnsembleVoter(build_default_engines()).fit(features, y)
    return builder, matches, y, voter


app = FastAPI(title="Football Prediction Engine", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Football Prediction Engine",
        "health": "/health",
        "predict": "/predict",
        "validate": "/validate",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    builder, matches, _, voter = _engine_context()
    context: dict[str, Any] = {
        "match_id": request.match_id,
        "odds_home_implied": request.odds_home_implied,
        "odds_draw_implied": request.odds_draw_implied,
        "odds_away_implied": request.odds_away_implied,
    }
    if request.season is not None:
        context["season"] = request.season
    match_features = builder.build_future_match(matches, request.home_team, request.away_team, **context)
    return voter.format_match_output(match_features)


@app.get("/validate")
def validate() -> dict[str, Any]:
    builder, matches, y, _ = _engine_context()
    features, _ = builder.transform(matches)
    return full_walk_forward_report(features, y)
