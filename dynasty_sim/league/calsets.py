"""Calibration leagues taken from the simulation itself (kept outside engine/, which must not name any group)."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import load_params
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.calibrate import LeagueSet
from dynasty_sim.league.schedule import season_schedule

HUMAN_ONLY = {"league.prospect_race_share": {"human": 1.0, "goliath": 0.0, "elf": 0.0},
              "demo.house_races": ["human"] * 8}


def build_loop_league_set(seed: int = 2024, n_seasons: int = 20, warmup: int = 10, overrides: dict | None = None) -> LeagueSet:
    """Calibration league taken from the simulation itself: the games a Human-only world actually played (real
    rosters, injuries and mid-season moves, coaches, schedules), for `n_seasons` seasons after a warm-up. Candidate
    engine parameters are scored by replaying exactly those matchups."""
    from dynasty_sim.history.sweep import _apply
    from dynasty_sim.league.world import World
    P = load_params()
    _apply(P, HUMAN_ONLY if overrides is None else overrides)
    w = World(seed, P, history=False)
    w.league.capture_games = []
    w.run(warmup + n_seasons)
    games = [g for g in w.league.capture_games if g[0] > warmup]
    years = sorted({g[0] for g in games})
    PL = np.concatenate([g[1] for g in games])
    NPL = np.concatenate([g[2] for g in games])
    CO = np.concatenate([g[3] for g in games])
    H = np.array([[p[0] for g in games if g[0] == y for p in g[4]] for y in years])
    A = np.array([[p[1] for g in games if g[0] == y for p in g[4]] for y in years])
    return LeagueSet(PL, NPL.astype(np.int64), CO, H, A, np.arange(len(PL), dtype=np.uint32) + 100_000,
                     len(years), P.league["n_teams"], P.league["games"])
