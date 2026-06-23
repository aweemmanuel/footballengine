from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


OUTCOMES = ["home_win", "draw", "away_win"]


@dataclass
class TeamHistory:
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    xg_for: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    xg_against: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    shots_for: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    points: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    home_points: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    away_points: deque[float] = field(default_factory=lambda: deque(maxlen=38))
    last_date: pd.Timestamp | None = None
    elo: float = 1500.0


def _mean(values: deque[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def _rolling(values: deque[float], window: int, fallback: float = 0.0) -> float:
    if not values:
        return fallback
    return float(np.mean(list(values)[-window:]))


def _weighted(values: deque[float], window: int, decay: float = 0.9, fallback: float = 0.0) -> float:
    if not values:
        return fallback
    arr = np.asarray(list(values)[-window:], dtype=float)
    weights = np.asarray([decay**i for i in range(len(arr))][::-1], dtype=float)
    return float(np.dot(arr, weights) / weights.sum())


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def _points_for(home_goals: int, away_goals: int) -> tuple[int, int]:
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


class FeatureBuilder:
    """Build chronological features using only information available before each match."""

    def __init__(self, windows: tuple[int, ...] = (5, 10, 15, 38), home_elo_advantage: float = 65.0):
        self.windows = windows
        self.home_elo_advantage = home_elo_advantage

    def transform(self, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = matches.sort_values("date").reset_index(drop=True).copy()
        histories: defaultdict[str, TeamHistory] = defaultdict(TeamHistory)
        h2h: defaultdict[tuple[str, str], deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=5))
        rows: list[dict[str, Any]] = []
        labels: list[str] = []

        for _, match in df.iterrows():
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_hist = histories[home]
            away_hist = histories[away]
            date = pd.Timestamp(match["date"])
            h2h_key = tuple(sorted((home, away)))
            prior_h2h = list(h2h[h2h_key])

            row: dict[str, Any] = {
                "match_id": match.get("match_id", ""),
                "date": match["date"],
                "season": int(match["season"]),
                "home_team": home,
                "away_team": away,
                "home_elo": home_hist.elo,
                "away_elo": away_hist.elo,
                "elo_diff": home_hist.elo + self.home_elo_advantage - away_hist.elo,
                "home_advantage": 1.0,
                "rest_days_diff": float(match.get("home_rest_days", 6) - match.get("away_rest_days", 6)),
                "temperature": float(match.get("temperature", 14.0)),
                "humidity": float(match.get("humidity", 60.0)),
                "wind_speed": float(match.get("wind_speed", 10.0)),
                "precipitation_probability": float(match.get("precipitation_probability", 0.25)),
                "referee_cards_per_match": float(match.get("referee_cards_per_match", 4.0)),
                "referee_fouls_per_match": float(match.get("referee_fouls_per_match", 22.0)),
                "referee_penalties_per_match": float(match.get("referee_penalties_per_match", 0.2)),
                "odds_home_implied": float(match.get("odds_home_implied", 0.40)),
                "odds_draw_implied": float(match.get("odds_draw_implied", 0.28)),
                "odds_away_implied": float(match.get("odds_away_implied", 0.32)),
                "odds_home_movement": float(match.get("odds_home_implied", 0.40) - match.get("odds_home_open", 0.40)),
                "odds_draw_movement": float(match.get("odds_draw_implied", 0.28) - match.get("odds_draw_open", 0.28)),
                "odds_away_movement": float(match.get("odds_away_implied", 0.32) - match.get("odds_away_open", 0.32)),
                "tipster_home_consensus": float(match.get("tipster_home_consensus", match.get("odds_home_implied", 0.40))),
                "tipster_draw_consensus": float(match.get("tipster_draw_consensus", match.get("odds_draw_implied", 0.28))),
                "tipster_away_consensus": float(match.get("tipster_away_consensus", match.get("odds_away_implied", 0.32))),
                "tipster_agreement": float(match.get("tipster_agreement", 0.5)),
                "match_importance": float(match.get("match_importance", 0.5)),
                "stadium_altitude": float(match.get("stadium_altitude", 50)),
            }

            for window in self.windows:
                row[f"home_xg_for_{window}"] = _rolling(home_hist.xg_for, window, 1.25)
                row[f"home_xga_{window}"] = _rolling(home_hist.xg_against, window, 1.25)
                row[f"away_xg_for_{window}"] = _rolling(away_hist.xg_for, window, 1.10)
                row[f"away_xga_{window}"] = _rolling(away_hist.xg_against, window, 1.25)
                row[f"xg_diff_{window}"] = row[f"home_xg_for_{window}"] - row[f"away_xg_for_{window}"]
                row[f"xga_diff_{window}"] = row[f"away_xga_{window}"] - row[f"home_xga_{window}"]
                row[f"home_form_{window}"] = _weighted(home_hist.points, window, 0.9, 1.2)
                row[f"away_form_{window}"] = _weighted(away_hist.points, window, 0.9, 1.1)
                row[f"form_diff_{window}"] = row[f"home_form_{window}"] - row[f"away_form_{window}"]

            row["home_home_form_5"] = _weighted(home_hist.home_points, 5, 0.9, 1.3)
            row["away_away_form_5"] = _weighted(away_hist.away_points, 5, 0.9, 1.0)
            row["venue_form_diff_5"] = row["home_home_form_5"] - row["away_away_form_5"]
            row["home_xg_conversion_10"] = _rolling(home_hist.goals_for, 10, 1.1) / max(row["home_xg_for_10"], 0.1)
            row["away_xg_conversion_10"] = _rolling(away_hist.goals_for, 10, 1.0) / max(row["away_xg_for_10"], 0.1)
            row["home_shooting_efficiency_10"] = row["home_xg_for_10"] / max(_rolling(home_hist.shots_for, 10, 10.0), 1.0)
            row["away_shooting_efficiency_10"] = row["away_xg_for_10"] / max(_rolling(away_hist.shots_for, 10, 9.0), 1.0)
            row["h2h_home_wins_5"] = float(sum(1 for item in prior_h2h if item["winner"] == home))
            row["h2h_away_wins_5"] = float(sum(1 for item in prior_h2h if item["winner"] == away))
            row["h2h_draws_5"] = float(sum(1 for item in prior_h2h if item["winner"] == "draw"))
            row["h2h_goal_diff_5"] = float(sum(item.get(home, 0.0) - item.get(away, 0.0) for item in prior_h2h))

            rows.append(row)
            labels.append(str(match["result"]))
            self._update_histories(histories, h2h, match)

        features = pd.DataFrame(rows)
        y = pd.Series(labels, name="result")
        return features, y

    def build_future_match(self, historical_matches: pd.DataFrame, home_team: str, away_team: str, **context: Any) -> pd.DataFrame:
        template = historical_matches.tail(1).copy()
        if template.empty:
            raise ValueError("At least one historical match is required before predicting a future match.")
        next_date = pd.Timestamp(template.iloc[-1]["date"]) + pd.Timedelta(days=3)
        row = template.iloc[-1].to_dict()
        row.update(
            {
                "match_id": context.pop("match_id", "future_match"),
                "date": context.pop("date", next_date.date().isoformat()),
                "season": int(context.pop("season", template.iloc[-1]["season"])),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": 0,
                "away_goals": 0,
                "home_xg": 0.0,
                "away_xg": 0.0,
                "result": "draw",
            }
        )
        row.update(context)
        combined = pd.concat([historical_matches, pd.DataFrame([row])], ignore_index=True)
        features, _ = self.transform(combined)
        return features.tail(1).reset_index(drop=True)

    def _update_histories(
        self,
        histories: defaultdict[str, TeamHistory],
        h2h: defaultdict[tuple[str, str], deque[dict[str, float]]],
        match: pd.Series,
    ) -> None:
        home = str(match["home_team"])
        away = str(match["away_team"])
        home_goals = int(match["home_goals"])
        away_goals = int(match["away_goals"])
        home_points, away_points = _points_for(home_goals, away_goals)
        home_hist = histories[home]
        away_hist = histories[away]

        home_hist.goals_for.append(home_goals)
        home_hist.goals_against.append(away_goals)
        home_hist.xg_for.append(float(match.get("home_xg", home_goals)))
        home_hist.xg_against.append(float(match.get("away_xg", away_goals)))
        home_hist.shots_for.append(float(match.get("home_shots", 10)))
        home_hist.points.append(home_points)
        home_hist.home_points.append(home_points)
        home_hist.last_date = pd.Timestamp(match["date"])

        away_hist.goals_for.append(away_goals)
        away_hist.goals_against.append(home_goals)
        away_hist.xg_for.append(float(match.get("away_xg", away_goals)))
        away_hist.xg_against.append(float(match.get("home_xg", home_goals)))
        away_hist.shots_for.append(float(match.get("away_shots", 9)))
        away_hist.points.append(away_points)
        away_hist.away_points.append(away_points)
        away_hist.last_date = pd.Timestamp(match["date"])

        home_score = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        expected_home = _expected_score(home_hist.elo + self.home_elo_advantage, away_hist.elo)
        change = 24.0 * (home_score - expected_home)
        home_hist.elo += change
        away_hist.elo -= change

        if home_goals > away_goals:
            winner = home
        elif home_goals < away_goals:
            winner = away
        else:
            winner = "draw"
        h2h[tuple(sorted((home, away)))].append({"winner": winner, home: home_goals, away: away_goals})
