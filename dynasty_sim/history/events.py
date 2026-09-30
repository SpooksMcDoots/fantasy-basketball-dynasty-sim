"""History events: everything worth retelling, with the context needed to chain it into arcs.

Each event stores (year, actors, houses, metric, value, percentile, prior-context ids). Person actors and houses are
"protagonists"; an event's context is the previous event of each of its protagonists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional

import numpy as np

# importance: 0 minor bookkeeping, 1 notable, 2 major, 3 landmark. Arcs are built from importance >= 1.
MINOR, NOTABLE, MAJOR, LANDMARK = range(4)


@dataclass
class HistoryEvent:
    id: int
    year: int
    kind: str
    actors: tuple = ()                # person ids
    houses: tuple = ()                # house ids
    importance: int = NOTABLE
    metric: str = ""
    value: Optional[float] = None
    percentile: Optional[float] = None
    context: tuple = ()               # ids of the previous event of each protagonist
    detail: dict = field(default_factory=dict)

    def protagonists(self) -> list:
        return [("person", a) for a in self.actors] + [("house", h) for h in self.houses]


class EventLog:
    def __init__(self) -> None:
        self.events: list[HistoryEvent] = []
        self._last: dict = {}

    def add(self, kind: str, year: int, actors=(), houses=(), importance: int = NOTABLE, metric: str = "",
            value: Optional[float] = None, percentile: Optional[float] = None, **detail) -> HistoryEvent:
        actors, houses = tuple(dict.fromkeys(actors)), tuple(dict.fromkeys(houses))
        ev = HistoryEvent(len(self.events), year, kind, actors, houses, importance, metric, value, percentile, (), detail)
        if importance >= NOTABLE:
            ev.context = tuple(dict.fromkeys(self._last[p] for p in ev.protagonists() if p in self._last))
            for p in ev.protagonists():
                self._last[p] = ev.id
        self.events.append(ev)
        return ev

    def by_kind(self, kind: str) -> list[HistoryEvent]:
        return [e for e in self.events if e.kind == kind]

    def notable(self) -> list[HistoryEvent]:
        return [e for e in self.events if e.importance >= NOTABLE]


def percentile_rank(value: float, samples) -> float:
    """Fraction of historical samples at or below `value` (0..1)."""
    s = np.asarray(samples, float)
    return float((s <= value).mean()) if len(s) else 0.5


def game_win_prob(mov_a: float, mov_b: float, home_edge: float = 2.5, sigma: float = 13.0) -> float:
    """Pregame win probability of team a hosting team b, from season margin-of-victory ratings."""
    return NormalDist().cdf((mov_a - mov_b + home_edge) / sigma)


def series_win_prob(p_game: float, best_of: int) -> float:
    """Probability of winning a best-of-n series with a constant per-game probability."""
    from math import comb
    need = best_of // 2 + 1
    return sum(comb(need - 1 + k, k) * p_game ** need * (1 - p_game) ** k for k in range(best_of - need + 1))
