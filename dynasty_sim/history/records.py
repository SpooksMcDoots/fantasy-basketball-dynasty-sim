"""Running game / season / career maxima and minima, per race and overall. A break is reported once per season."""
from __future__ import annotations

from dataclasses import dataclass

PLAYER_GAME = ("pts", "reb", "ast", "stl", "blk", "tpm")
PLAYER_SEASON = ("ppg", "rpg", "apg", "spg", "bpg", "value")
PLAYER_CAREER = ("pts", "reb", "ast", "stl", "blk", "games", "value")
MIN_GAMES_FOR_SEASON_RECORD = 40


@dataclass(frozen=True)
class RecordEntry:
    value: float
    holder: int              # person id for player records, team id for team records
    year: int


@dataclass(frozen=True)
class RecordBreak:
    kind: str                # game | season | career | team_game | team_season
    stat: str
    scope: str               # "all" or a race name
    direction: str           # max | min
    new: RecordEntry
    old: RecordEntry


class RecordBook:
    """`best` holds the official records through last season; `pending` this season's candidates."""

    def __init__(self) -> None:
        self.best: dict = {}
        self.pending: dict = {}

    def offer(self, kind: str, stat: str, value: float, holder: int, year: int, scopes=("all",), direction: str = "max") -> None:
        for scope in scopes:
            key = (kind, stat, scope, direction)
            cur = self.pending.get(key)
            if cur is None or (value > cur.value if direction == "max" else value < cur.value):
                self.pending[key] = RecordEntry(float(value), int(holder), int(year))

    def finalize(self, year: int) -> list[RecordBreak]:
        """Fold this season's candidates into the books; return the genuine breaks (not first-time baselines)."""
        breaks = []
        for key, cand in self.pending.items():
            kind, stat, scope, direction = key
            old = self.best.get(key)
            if old is None:
                self.best[key] = cand
            elif (cand.value > old.value) if direction == "max" else (cand.value < old.value):
                breaks.append(RecordBreak(kind, stat, scope, direction, cand, old))
                self.best[key] = cand
        self.pending = {}
        return breaks

    def holder(self, kind: str, stat: str, scope: str = "all", direction: str = "max"):
        return self.best.get((kind, stat, scope, direction))
