"""Hall of Fame: career value above a floor, paced to ~1.25 inductees per season, plus a wing for dynasty builders."""
from __future__ import annotations

import numpy as np

RATE_PER_SEASON = 1.25
MIN_SEASONS = 5
WAIT_YEARS = 1                  # a player is eligible the year after retiring
MIN_RETIRED_FOR_FLOOR = 20
FLOOR_QUANTILE = 0.70           # among retired careers with enough seasons
DYNASTY_TITLES = 3              # titles won during a head's tenure to earn the contributor wing


class HallOfFame:
    def __init__(self) -> None:
        self.players: dict[int, int] = {}          # pid -> induction year
        self.builders: dict[int, tuple] = {}       # pid -> (year, house_id, titles)
        self._budget = 0.0

    def inductees_this_year(self, year: int, careers: dict) -> list[int]:
        """Pick this season's player inductees: best eligible careers, paced to the target rate."""
        self._budget = min(self._budget + RATE_PER_SEASON, 2.5)
        elig = [c for c in careers.values() if c.retired_year is not None and year - c.retired_year >= WAIT_YEARS
                and c.pid not in self.players and len(c.seasons) >= MIN_SEASONS]
        retired = [c.career_value() for c in careers.values() if c.retired_year is not None and len(c.seasons) >= MIN_SEASONS]
        floor = float(np.quantile(retired, FLOOR_QUANTILE)) if len(retired) >= MIN_RETIRED_FOR_FLOOR else float("inf")
        cands = sorted((c for c in elig if c.career_value() >= floor), key=lambda c: (-c.career_value(), c.pid))
        picked = []
        while self._budget >= 1.0 and cands:
            c = cands.pop(0)
            self.players[c.pid] = year
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
