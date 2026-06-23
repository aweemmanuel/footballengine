from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TEAMS = [
    "Arsenal",
    "Manchester City",
    "Liverpool",
    "Chelsea",
    "Tottenham",
    "Manchester United",
    "Newcastle",
    "Aston Villa",
]


@dataclass(frozen=True)
class MatchSchema:
    date: str = "date"
    season: str = "season"
    home_team: str = "home_team"
    away_team: str = "away_team"
    home_goals: str = "home_goals"
    away_goals: str = "away_goals"


def result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def generate_sample_matches(rows: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strengths = {team: rng.normal(0.0, 0.55) for team in TEAMS}
    attack = {team: 1.2 + strengths[team] * 0.35 + rng.normal(0, 0.08) for team in TEAMS}
    defense = {team: 1.1 - strengths[team] * 0.25 + rng.normal(0, 0.08) for team in TEAMS}
    start = pd.Timestamp("2017-08-12")
    records = []

    for i in range(rows):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        season = 2017 + (i // max(1, rows // 8))
        date = start + pd.Timedelta(days=int(i * 5 + rng.integers(0, 3)))
        home_edge = 0.28
        home_xg = max(0.15, attack[home] + defense[away] * 0.35 + home_edge + rng.normal(0, 0.22))
        away_xg = max(0.15, attack[away] + defense[home] * 0.35 + rng.normal(0, 0.22))
        home_goals = int(rng.poisson(home_xg))
        away_goals = int(rng.poisson(away_xg))
        home_rest = int(rng.integers(3, 9))
        away_rest = int(rng.integers(3, 9))
        temp = float(rng.normal(14, 7))
        humidity = float(np.clip(rng.normal(62, 15), 25, 98))
        wind = float(np.clip(rng.normal(12, 5), 0, 35))
        rain = float(np.clip(rng.beta(2, 5), 0, 1))
        ref_cards = float(np.clip(rng.normal(4.1, 0.9), 1.2, 7.8))
        ref_fouls = float(np.clip(rng.normal(22, 5), 10, 38))
        ref_penalties = float(np.clip(rng.normal(0.24, 0.08), 0.02, 0.6))

        # Odds are noisy but correlated with latent strength and home advantage.
        home_logit = strengths[home] - strengths[away] + home_edge + rng.normal(0, 0.18)
        away_logit = strengths[away] - strengths[home] - home_edge + rng.normal(0, 0.18)
        raw = np.exp([home_logit, 0.0, away_logit])
        probs = raw / raw.sum()
        tipster_noise = rng.normal(0, 0.035, size=3)
        tipster = np.clip(probs + tipster_noise, 0.04, 0.9)
        tipster = tipster / tipster.sum()

        records.append(
            {
                "match_id": f"EPL_{season}_{i + 1:04d}",
                "date": date.date().isoformat(),
                "season": season,
                "league": "EPL",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": round(home_xg, 3),
                "away_xg": round(away_xg, 3),
                "home_shots": int(max(4, rng.poisson(home_xg * 7))),
                "away_shots": int(max(4, rng.poisson(away_xg * 7))),
                "home_rest_days": home_rest,
                "away_rest_days": away_rest,
                "temperature": round(temp, 1),
                "humidity": round(humidity, 1),
                "wind_speed": round(wind, 1),
                "precipitation_probability": round(rain, 3),
                "referee_cards_per_match": round(ref_cards, 2),
                "referee_fouls_per_match": round(ref_fouls, 2),
                "referee_penalties_per_match": round(ref_penalties, 3),
                "odds_home_implied": round(float(probs[0]), 4),
                "odds_draw_implied": round(float(probs[1]), 4),
                "odds_away_implied": round(float(probs[2]), 4),
                "odds_home_open": round(float(np.clip(probs[0] + rng.normal(0, 0.025), 0.05, 0.9)), 4),
                "odds_draw_open": round(float(np.clip(probs[1] + rng.normal(0, 0.025), 0.05, 0.9)), 4),
                "odds_away_open": round(float(np.clip(probs[2] + rng.normal(0, 0.025), 0.05, 0.9)), 4),
                "tipster_home_consensus": round(float(tipster[0]), 4),
                "tipster_draw_consensus": round(float(tipster[1]), 4),
                "tipster_away_consensus": round(float(tipster[2]), 4),
                "tipster_agreement": round(float(1.0 - np.var(tipster) * 6), 4),
                "match_importance": round(float(rng.uniform(0.25, 1.0)), 3),
                "stadium_altitude": int(rng.integers(5, 260)),
            }
        )

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["result"] = [result_label(h, a) for h, a in zip(df.home_goals, df.away_goals)]
    return df


def load_matches(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        return generate_sample_matches()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    if "result" not in df.columns and {"home_goals", "away_goals"}.issubset(df.columns):
        df["result"] = [result_label(h, a) for h, a in zip(df.home_goals, df.away_goals)]
    return df.sort_values("date").reset_index(drop=True)


def save_matches(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
