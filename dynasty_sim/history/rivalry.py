"""Rivalry index between houses: accumulates from games, playoffs, marriages, disputes and moving players; decays yearly."""
from __future__ import annotations

from collections import defaultdict

PER_GAME = 1.0
CLOSE_GAME = 2.0            # extra, for a game decided by <= 3
CLOSE_MARGIN = 3
PLAYOFF_SERIES = 5.0
MARRIAGE_OR_DISPUTE = 8.0
PLAYER_MOVE = 4.0
DECAY = 0.85
RELATIVE_THRESHOLD = 1.07   # a pairing is a rivalry when its index is this multiple of the median pair's
MIN_LEVEL = 100.0           # ...and at least this high (guards the early seasons)
SUSTAINED = 3               # ...for this many consecutive seasons


def pair(a: int, b: int) -> tuple:
    return (a, b) if a < b else (b, a)


class RivalryBoard:
    def __init__(self, relative: float = RELATIVE_THRESHOLD) -> None:
        self.index: dict[tuple, float] = defaultdict(float)
        self.relative = relative
        self.level = MIN_LEVEL            # absolute index a pair must reach this season
        self.peak: dict[tuple, tuple] = {}            # pair -> (peak index, year)
        self.flagged: dict[tuple, int] = {}           # pair -> year the rivalry was established
        self.streak: dict[tuple, int] = defaultdict(int)
        self.reasons: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))

    def add(self, a: int, b: int, points: float, reason: str) -> None:
        if a != b:
            k = pair(a, b)
            self.index[k] += points
            self.reasons[k][reason] += points

    def game(self, a: int, b: int, margin: int) -> None:
        self.add(a, b, PER_GAME + (CLOSE_GAME if abs(margin) <= CLOSE_MARGIN else 0.0), "games")

    def playoff_series(self, a: int, b: int) -> None:
        self.add(a, b, PLAYOFF_SERIES, "playoffs")

    def marriage_or_dispute(self, a: int, b: int, what: str) -> None:
        self.add(a, b, MARRIAGE_OR_DISPUTE, what)

    def player_move(self, a: int, b: int) -> None:
        self.add(a, b, PLAYER_MOVE, "moves")

    def end_season(self, year: int) -> list[tuple]:
        """Decay, then return the pairs that newly crossed the threshold this season."""
        for k in list(self.index):
            self.index[k] *= DECAY
            if self.index[k] > self.peak.get(k, (0.0, 0))[0]:
                self.peak[k] = (self.index[k], year)
        import numpy as np
        self.level = max(MIN_LEVEL, self.relative * float(np.median(list(self.index.values())))) if self.index else MIN_LEVEL
        fresh = []
        for k, v in self.index.items():
            self.streak[k] = self.streak[k] + 1 if v >= self.level else 0
            if self.streak[k] >= SUSTAINED and k not in self.flagged:
                self.flagged[k] = year
                fresh.append(k)
        return fresh

    def why(self, k: tuple, n: int = 2) -> list[str]:
        return [r for r, _ in sorted(self.reasons[k].items(), key=lambda kv: -kv[1])[:n]]

    def top(self, n: int = 3) -> list[tuple]:
        return sorted(self.index.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
