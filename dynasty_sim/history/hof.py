"""Hall of Fame: peak and honours count as much as longevity, paced to ~1.25 inductees a season, plus a builders' wing.

A retired career is scored as the average of three standardised parts, so a long career alone cannot carry a player:
  * career value      total points added over replacement (rewards longevity)
  * best-five value   the five best seasons added together (rewards peak)
  * honours           awards and titles won as a core player (rewards how the player was seen and what he won)

Each part is standardised against the career's own role (guard, forward or center, by height), so a center is judged
against centers and a guard against guards. Box value is not comparable across roles (a center's rebounds and blocks
score very differently from a guard's assists), and pooling them filled the Hall with whichever role the formula favours.
"""
from __future__ import annotations

import numpy as np

RATE_PER_SEASON = 1.25
MIN_SEASONS = 5
WAIT_YEARS = 1                  # a player is eligible the year after retiring
MIN_RETIRED_FOR_FLOOR = 20
FLOOR_QUANTILE = 0.70           # among retired careers with enough seasons
DYNASTY_TITLES = 3              # titles won during a head's tenure to earn the contributor wing
PEAK_SEASONS = 5
MIN_GROUP = 8                   # a role with fewer retired careers than this is standardised against everyone
TITLE_POINTS = 3
HONOUR_POINTS = {"MVP": 10, "Finals MVP": 4, "Defensive Player": 4, "Scoring Champion": 3, "All-League": 3,
                 "Rookie of the Year": 1, "Most Improved": 1}


def peak_value(career, n: int = PEAK_SEASONS) -> float:
    return float(sum(sorted((s.value for s in career.seasons), reverse=True)[:n]))


def honour_points(career, titles: dict) -> float:
    return float(sum(HONOUR_POINTS.get(label, 0) for _, label in career.awards) + TITLE_POINTS * titles.get(career.pid, 0))


def _z(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)


def _z_by_role(values: np.ndarray, roles: list) -> np.ndarray:
    """Standardise within each role that has enough members; everyone else against the whole pool."""
    out = _z(values)
    for role in {r for r in roles if r is not None}:
        idx = np.array([i for i, r in enumerate(roles) if r == role])
        if len(idx) >= MIN_GROUP:
            out[idx] = _z(values[idx])
    return out


def score_careers(careers: list, titles: dict, roles: dict | None = None) -> dict[int, dict]:
    """pid -> {career, peak, honours, score, role}: the score is the mean of the three role-relative standardised parts."""
    if not careers:
        return {}
    parts = np.array([[c.career_value(), peak_value(c), honour_points(c, titles)] for c in careers])
    role_list = [(roles or {}).get(c.pid) for c in careers]
    z = np.column_stack([_z_by_role(parts[:, i], role_list) for i in range(3)])
    return {c.pid: {"career": float(p[0]), "peak": float(p[1]), "honours": float(p[2]), "score": float(zs.mean()), "role": r}
            for c, p, zs, r in zip(careers, parts, z, role_list)}


class HallOfFame:
    def __init__(self) -> None:
        self.players: dict[int, int] = {}          # pid -> induction year
        self.record: dict[int, dict] = {}          # pid -> the score parts at induction
        self.builders: dict[int, tuple] = {}       # pid -> (year, house_id, titles)
        self._budget = 0.0

    def inductees_this_year(self, year: int, careers: dict, titles: dict | None = None, roles: dict | None = None) -> list[int]:
        """Pick this season's player inductees: best-scoring eligible careers, paced to the target rate."""
        self._budget = min(self._budget + RATE_PER_SEASON, 2.5)
        retired = [c for c in careers.values() if c.retired_year is not None and len(c.seasons) >= MIN_SEASONS]
        scores = score_careers(retired, titles or {}, roles)
        floor = (float(np.quantile([v["score"] for v in scores.values()], FLOOR_QUANTILE))
                 if len(retired) >= MIN_RETIRED_FOR_FLOOR else float("inf"))
        cands = sorted((c for c in retired if year - c.retired_year >= WAIT_YEARS and c.pid not in self.players
                        and scores[c.pid]["score"] >= floor), key=lambda c: (-scores[c.pid]["score"], c.pid))
        picked = []
        while self._budget >= 1.0 and cands:
            c = cands.pop(0)
            self.players[c.pid] = year
            self.record[c.pid] = scores[c.pid]
            picked.append(c.pid)
            self._budget -= 1.0
        return picked

    def builders_this_year(self, year: int, tenures: list) -> list[tuple]:
        """tenures: (head pid, house id, first year, last year, titles) for heads whose tenure has ended."""
        out = []
        for pid, hid, y0, y1, titles in tenures:
            if pid not in self.builders and pid not in self.players and titles >= DYNASTY_TITLES:
                self.builders[pid] = (year, hid, titles)
                out.append((pid, hid, titles))
        return out
