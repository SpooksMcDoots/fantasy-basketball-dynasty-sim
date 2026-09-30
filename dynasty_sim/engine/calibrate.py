"""Calibration against the validation table: fixed league set, metrics, loss, optimiser.

The league set (players, schedules, seeds) is built once from fixed seeds, so the loss is a deterministic
function of the engine parameters. Only the single-population calibration league is ever used for fitting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dynasty_sim.config import Params, load_params
from dynasty_sim.core.reference import Reference, frozen_reference
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.interface import CoachParams, NumbaEngine
from dynasty_sim.engine.params import load_engine_params
from dynasty_sim.league.pool import random_teams, selected_units
from dynasty_sim.league.schedule import season_schedule
from dynasty_sim.engine.composites import build_players

CONFIG = Path(__file__).parent.parent / "config"
CALIBRATED = CONFIG / "engine_calibrated.yaml"


def validation_table() -> dict:
    return yaml.safe_load((CONFIG / "validation.yaml").read_text())["metrics"]


@dataclass
class LeagueSet:
    PL: np.ndarray
    NPL: np.ndarray
    CO: np.ndarray
    H: np.ndarray            # [S, Gs] home team index within the season
    A: np.ndarray
    seeds: np.ndarray
    n_seasons: int
    n_teams: int
    games: int


def build_league_set(P: Params, ref: Reference, n_seasons: int = 20, n_teams: int = 8, games: int = 120,
                     seed: int = 2024, coach: CoachParams | None = None) -> LeagueSet:
    rng = np.random.default_rng(seed)
    co_row = (coach or CoachParams()).row()
    PLs, Hs, As = [], [], []
    for s in range(n_seasons):
        rows = build_players(selected_units(n_teams * 13, rng, P, ref), ref)
        teams = random_teams(rows, n_teams, rng)
        h, a = season_schedule(n_teams, games, rng, s)
        PLs.append(np.stack([teams[h], teams[a]], axis=1))
        Hs.append(h)
        As.append(a)
    PL = np.concatenate(PLs)
    G = len(PL)
    return LeagueSet(PL, np.full((G, 2), 13, np.int64), np.tile(co_row, (G, 2, 1)), np.array(Hs), np.array(As),
                     np.arange(G, dtype=np.uint32) + 100_000, n_seasons, n_teams, games)


def compute_metrics(BOX, SCORE, POSS, ls: LeagueSet) -> dict:
    Sn, Gs = ls.n_seasons, ls.H.shape[1]
    tm = BOX.sum(2).reshape(Sn, Gs, 2, S.NSTAT)
    sc = SCORE.reshape(Sn, Gs, 2).astype(float)
    po = POSS.reshape(Sn, Gs, 2).astype(float)
    c = lambda k: tm[..., k]
    elapsed = c(S.ST_SEC) / 5.0
    fga, fta, fgm, tpm, tpa, ftm = c(S.ST_FGA), c(S.ST_FTA), c(S.ST_FGM), c(S.ST_TPM), c(S.ST_TPA), c(S.ST_FTM)
    tov, orb, drb = c(S.ST_TOV), c(S.ST_ORB), c(S.ST_DRB)
    per_season = lambda x: x.reshape(Sn, -1).sum(1)          # sum over games and both teams
    m = {
        "pace": float(np.mean(per_season(po * 2880.0 / elapsed) / (2 * Gs))),
        "ortg": float(np.mean(100 * per_season(sc) / per_season(po))),
        "ppg": float(sc.mean()),
        "efg": float(np.mean((per_season(fgm) + 0.5 * per_season(tpm)) / per_season(fga))),
        "ts": float(np.mean(per_season(sc) / (2 * (per_season(fga) + 0.44 * per_season(fta))))),
        "tov_pct": float(100 * np.mean(per_season(tov) / (per_season(fga) + 0.44 * per_season(fta) + per_season(tov)))),
        "orb_pct": float(100 * np.mean(per_season(orb) / (per_season(orb) + per_season(drb)))),
        "fta_fga": float(np.mean(per_season(fta) / per_season(fga))),
        "tpa_fga": float(np.mean(per_season(tpa) / per_season(fga))),
        "tp_pct": float(np.mean(per_season(tpm) / per_season(tpa))),
        "ft_pct": float(np.mean(per_season(ftm) / per_season(fta))),
        "ast": float(c(S.ST_AST).mean()), "stl": float(c(S.ST_STL).mean()),
        "blk": float(c(S.ST_BLK).mean()), "pf": float(c(S.ST_PF).mean()),
    }
    margin = sc[..., 0] - sc[..., 1]
    m["home_margin"] = float(margin.mean())
    m["home_win"] = float((margin > 0).mean())
    m["margin_sd_100"] = float(np.mean((margin / po.mean(2) * 100).std(1)))
    best, sd_team, win_sd = [], [], []
    for s in range(Sn):
        mov, scores, wins = [], [], []
        for t in range(ls.n_teams):
            hm, am = ls.H[s] == t, ls.A[s] == t
            played = hm | am
            team_margin = np.where(hm, margin[s], -margin[s])[played]
            team_score = np.where(hm, sc[s, :, 0], sc[s, :, 1])[played]
            mov.append(team_margin.mean())
            scores.append(team_score.std())
            wins.append((team_margin > 0).astype(float))
        best.append(max(mov))
        sd_team.append(np.mean(scores))
        w = np.array([x[: len(x) // 30 * 30].reshape(-1, 30).mean(1) for x in wins])   # [teams, windows]
        win_sd.append(w.std(0).mean())
    m["best_mov"] = float(np.mean(best))
    m.update(_player_metrics(BOX, ls))
    m["team_score_sd"] = float(np.mean(sd_team))
    m["win_sd_30"] = float(np.mean(win_sd))
    return m


def _player_metrics(BOX, ls: "LeagueSet") -> dict:
    """Minutes and scoring concentration: how starters are used and how big individual games get."""
    Sn, Gs = ls.n_seasons, ls.H.shape[1]
    box = BOX.reshape(Sn, Gs, 2, S.MAX_ROSTER, S.NSTAT)
    starters = np.sort(box[..., S.ST_SEC], axis=3)[..., -5:].mean() / 60.0
    top = []
    for s in range(Sn):
        best = 0.0
        for t in range(ls.n_teams):
            hm, am = ls.H[s] == t, ls.A[s] == t
            pts = box[s][hm, 0, :, S.ST_PTS].sum(0) + box[s][am, 1, :, S.ST_PTS].sum(0)
            played = (box[s][hm, 0, :, S.ST_SEC] > 0).sum(0) + (box[s][am, 1, :, S.ST_SEC] > 0).sum(0)
            ok = played >= 40
            if ok.any():
                best = max(best, float((pts[ok] / played[ok]).max()))
        top.append(best)
    return {"starter_mpg": float(starters), "top_scorer_ppg": float(np.mean(top)),
            "max_game_pts": float(box[..., S.ST_PTS].max())}


def run_league(eng: NumbaEngine, ls: LeagueSet, prm: np.ndarray | None = None) -> dict:
    if prm is not None:
        eng.prm = prm
    BOX, SCORE, POSS = eng.simulate_arrays(ls.PL, ls.NPL, ls.CO, ls.seeds)
    return compute_metrics(BOX, SCORE, POSS, ls)


def metric_loss(m: dict, table: dict | None = None, detail: bool = False):
    """L = sum(((sim - mid) / half_width)^2); one-sided ceilings only penalise the excess."""
    table = table or validation_table()
    parts = {}
    for k, (lo, hi) in table.items():
        if lo is None:
            parts[k] = max(0.0, m[k] - hi) ** 2
        else:
            parts[k] = ((m[k] - 0.5 * (lo + hi)) / (0.5 * (hi - lo))) ** 2
    return parts if detail else float(sum(parts.values()))


def out_of_range(m: dict, table: dict | None = None) -> list[str]:
    table = table or validation_table()
    return [k for k, (lo, hi) in table.items() if (lo is not None and m[k] < lo) or m[k] > hi]


# ---- optimisation -------------------------------------------------------------------------
_LOGIT = {"ft", "tov", "make_rim", "make_mid", "make_three", "blk_rim", "blk_mid", "blk_three",
          "sf_rim", "sf_mid", "sf_three", "orb", "ast_rim", "ast_mid", "ast_three", "nsf_base"}
_LOG = {"pace_mean", "tau", "energy_drain", "home_foul", "beta_touch_rim", "beta_touch_mid", "beta_touch_three",
        "beta_c_rim", "beta_c_mid", "beta_c_three", "c_reach_rim", "c_reach_mid", "c_reach_three",
        "c_quick_rim", "c_quick_mid", "c_quick_three", "blk_slope", "size_rim", "sf_str", "sf_quick",
        "tov_slope", "orb_slope", "pace_game_sd", "shoot_game_sd", "foul_game_sd", "pace_shape", "score_flow", "reach_sat", "z_sat", "usage_sat", "bench_rec"}


def _load_defaults() -> dict:
    return yaml.safe_load((CONFIG / "engine.yaml").read_text())["params"]


def _encode(name: str, v: float) -> float:
    return float(np.log(v / (1 - v))) if name in _LOGIT else float(np.log(v)) if name in _LOG else float(v)


def _decode(name: str, x: float) -> float:
    x = float(np.clip(x, -9.0, 9.0))     # keeps probabilities strictly inside (0, 1) and slopes finite
    return float(1 / (1 + np.exp(-x))) if name in _LOGIT else float(np.exp(x)) if name in _LOG else float(x)


STAGE1 = ("make_rim", "make_mid", "make_three", "tov", "orb", "sf_rim", "sf_mid", "sf_three", "pace_mean",
          "nsf_base", "home_make", "home_foul")
STAGE2 = STAGE1 + ("beta_touch_rim", "beta_touch_mid", "beta_touch_three", "beta_c_rim", "beta_c_mid",
                   "beta_c_three", "blk_rim", "blk_mid", "blk_three", "tau", "ast_rim", "ast_mid", "ast_three",
                   "energy_drain", "steal_share", "ft", "c_reach_rim", "c_reach_mid", "c_reach_three",
                   "c_quick_rim", "c_quick_mid", "c_quick_three", "blk_slope", "size_rim", "sf_str",
                   "sf_quick", "tov_slope", "orb_slope", "pace_game_sd", "shoot_game_sd", "foul_game_sd", "pace_shape", "score_flow", "reach_sat", "z_sat", "usage_sat", "bench_rec", "sub_energy", "ev_mid", "ev_three")


_SLOPES = ("beta_touch_rim", "beta_touch_mid", "beta_touch_three", "beta_c_rim", "beta_c_mid", "beta_c_three",
           "c_reach_rim", "c_reach_mid", "c_reach_three", "c_quick_rim", "c_quick_mid", "c_quick_three",
           "blk_slope", "size_rim", "sf_str", "sf_quick", "tov_slope", "orb_slope")

# Physical-meaning bounds (decoded space). Slopes stay within x0.4..x2.5 of the spec starting values so the
# fitted engine keeps the mechanisms (contest geometry, size at the rim) instead of collapsing them.
FIXED_BOUNDS = {
    "make_rim": (0.30, 0.85), "make_mid": (0.25, 0.75), "make_three": (0.20, 0.55),
    "tov": (0.08, 0.20), "orb": (0.15, 0.35), "nsf_base": (0.02, 0.12), "ft": (0.70, 0.85),
    "ast_rim": (0.20, 0.90), "ast_mid": (0.20, 0.90), "ast_three": (0.20, 0.95),
    "pace_mean": (12.5, 17.5), "tau": (0.05, 0.40), "energy_drain": (0.001, 0.02),
    "home_foul": (1.0, 1.25), "steal_share": (0.40, 0.80), "home_make": (0.0, 0.12),
    "ev_mid": (-0.15, 0.15), "ev_three": (-0.15, 0.15),
    "pace_game_sd": (0.005, 0.15), "shoot_game_sd": (0.005, 0.20), "foul_game_sd": (0.01, 0.40),
    "pace_shape": (3.0, 40.0), "score_flow": (0.02, 1.0), "reach_sat": (8.0, 40.0), "z_sat": (1.0, 4.0), "usage_sat": (0.6, 3.0), "bench_rec": (0.004, 0.03), "sub_energy": (0.55, 0.85),
}


def param_bounds(name: str, defaults: dict) -> tuple[float, float]:
    if name in FIXED_BOUNDS:
        return FIXED_BOUNDS[name]
    d = defaults[name]
    if name in _SLOPES:
        return 0.4 * d, 2.5 * d
    return 0.3 * d, 3.0 * d                     # sf_*, blk_*: rates that may move about 3x either way


def calibrate(eng: NumbaEngine, ls: LeagueSet, names=STAGE2, start: dict | None = None, max_evals: int = 600,
              verbose: bool = True, bounded: bool = True):
    from scipy.optimize import minimize
    ref = eng.ref
    defaults = _load_defaults()
    cur = {**defaults, **(start or {})}
    lohi = [param_bounds(n, defaults) for n in names]
    bounds = [(_encode(n, lo), _encode(n, hi)) for n, (lo, hi) in zip(names, lohi)]
    x0 = np.array([np.clip(_encode(n, cur[n]), b[0], b[1]) for n, b in zip(names, bounds)])
    table = validation_table()
    count = [0]

    def f(x):
        ov = {n: _decode(n, v) for n, v in zip(names, x)}
        m = run_league(eng, ls, load_engine_params(ref, ov, calibrated=False))
        count[0] += 1
        val = metric_loss(m, table)
        if verbose and count[0] % 25 == 0:
            print(f"  eval {count[0]:4d} loss {val:8.2f}  out of range: {out_of_range(m, table)}", flush=True)
        return val

    res = minimize(f, x0, method="Powell", bounds=bounds if bounded else None,
                   options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-4})
    return {n: _decode(n, v) for n, v in zip(names, res.x)}, float(res.fun)


def config_hash(params: dict, ref: Reference) -> str:
    blob = json.dumps({k: round(float(v), 6) for k, v in sorted(params.items())}) + ref.fingerprint
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_calibrated(params: dict, loss: float, ref: Reference) -> None:
    doc = {"meta": {"config_hash": config_hash(params, ref), "reference": ref.fingerprint, "loss": round(loss, 4)},
           "params": {k: round(float(v), 6) for k, v in sorted(params.items())}}
    CALIBRATED.write_text("# Frozen output of dynasty_sim.engine.calibrate. Do not hand-edit; rerun calibration.\n"
                          + yaml.safe_dump(doc, sort_keys=False))
