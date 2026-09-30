"""Data-only engine interface: arrays in, results out, so a different engine can be swapped in later."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from dynasty_sim.core.reference import Reference, frozen_reference
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.params import load_engine_params


@dataclass
class CoachParams:
    style_bias: tuple = (0.0, 0.0, 0.0)   # EV bonus (points) for rim / mid / three
    pace_pref: float = 0.0                # +1 faster, -1 slower
    rotation_depth: int = 10
    foul_aggression: float = 0.0

    def row(self) -> np.ndarray:
        r = np.zeros(S.NCO)
        r[S.CO_BIAS_RIM], r[S.CO_BIAS_MID], r[S.CO_BIAS_THREE] = self.style_bias
        r[S.CO_PACE], r[S.CO_DEPTH], r[S.CO_FOUL] = self.pace_pref, self.rotation_depth, self.foul_aggression
        return r


@dataclass
class TeamInput:
    players: np.ndarray                   # [n<=13, NF] rows from composites.build_players
    coach: CoachParams = field(default_factory=CoachParams)


@dataclass
class GameInput:
    home: TeamInput
    away: TeamInput


@dataclass
class GameResult:
    score: np.ndarray                     # [2] home, away
    box: np.ndarray                       # [2, 13, NSTAT]
    poss: np.ndarray                      # [2]
    periods: int = 4
    log: Optional[np.ndarray] = None      # [n, NLOG] float32


class GameEngine(Protocol):
    def simulate(self, gi: GameInput, seed: int) -> GameResult: ...


def _pack(gis: list[GameInput]):
    G = len(gis)
    PL = np.zeros((G, 2, S.MAX_ROSTER, S.NF))
    NPL = np.zeros((G, 2), np.int64)
    CO = np.zeros((G, 2, S.NCO))
    for g, gi in enumerate(gis):
        for t, team in enumerate((gi.home, gi.away)):
            n = len(team.players)
            if n > S.MAX_ROSTER or int((team.players[:, S.C_AVAIL] > 0.5).sum()) < 5:
                raise ValueError("team needs 5..13 rows with at least 5 available")
            PL[g, t, :n] = team.players
            NPL[g, t] = n
            CO[g, t] = team.coach.row()
    return PL, NPL, CO


class NumbaEngine:
    def __init__(self, ref: Reference | None = None, overrides: dict | None = None):
        from dynasty_sim.engine import kernel          # deferred: importing compiles nothing until first call
        self._k = kernel
        self.ref = ref or frozen_reference()
        self.prm = load_engine_params(self.ref, overrides)

    def simulate(self, gi: GameInput, seed: int, want_log: bool = False) -> GameResult:
        PL, NPL, CO = _pack([gi])
        box = np.zeros((2, S.MAX_ROSTER, S.NSTAT))
        score = np.zeros(2, np.int64)
        poss = np.zeros(2, np.int64)
        log = np.zeros((600 if want_log else 0, S.NLOG), np.float32)
        n, periods = self._k.sim_game_single(PL[0], NPL[0], CO[0], self.prm, np.uint32(seed), box, score, poss, log)
        return GameResult(score, box, poss, periods + 1, log[:n] if want_log else None)

    def simulate_arrays(self, PL, NPL, CO, seeds):
        G = PL.shape[0]
        BOX = np.zeros((G, 2, S.MAX_ROSTER, S.NSTAT))
        SCORE = np.zeros((G, 2), np.int64)
        POSS = np.zeros((G, 2), np.int64)
        self._k.sim_games(PL, NPL, CO, self.prm, np.asarray(seeds, np.uint32), BOX, SCORE, POSS)
        return BOX, SCORE, POSS

    def simulate_batch(self, gis: list[GameInput], seeds) -> list[GameResult]:
        PL, NPL, CO = _pack(gis)
        BOX, SCORE, POSS = self.simulate_arrays(PL, NPL, CO, seeds)
        return [GameResult(SCORE[g], BOX[g], POSS[g]) for g in range(len(gis))]
