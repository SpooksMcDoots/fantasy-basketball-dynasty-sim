"""Measurements behind the M8 acceptance tests: team style vs roster ancestry, and multi-seed stability runs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dynasty_sim.config import load_params
from dynasty_sim.engine import schema as S
from dynasty_sim.league.world import World

CHECKED = ("pace", "ortg", "ppg", "efg", "ts", "tov_pct", "orb_pct", "fta_fga", "tpa_fga", "tp_pct", "ft_pct", "ast", "stl",
           "blk", "pf", "home_margin", "home_win", "best_mov")        # table rows computable from a season's box totals
STYLE = ("tpa_rate", "rim_share", "orb_pct", "tov_pct", "pace", "blk_pg")


def team_style(res) -> np.ndarray:
    """[n_teams, 6] style vector per team for one season, from its regular-season box totals."""
    T, O, poss, g = res.team_box, res.opp_box, res.team_poss, res.team_games
    fga = T[:, S.ST_FGA]
    elapsed = T[:, S.ST_SEC] / 5.0 / g                                  # seconds per game
    return np.column_stack([
        T[:, S.ST_TPA] / fga,
        T[:, S.ST_RIM] / fga,
        T[:, S.ST_ORB] / (T[:, S.ST_ORB] + O[:, S.ST_DRB]),
        T[:, S.ST_TOV] / (fga + 0.44 * T[:, S.ST_FTA] + T[:, S.ST_TOV]),
        poss / g * 2880.0 / elapsed,
        T[:, S.ST_BLK] / g,
    ])


def team_ancestry(res, rng: np.random.Generator | None = None) -> np.ndarray:
    """[n_teams, 3] minutes-weighted mean ancestry of each team's players.

    With `rng`, ancestry labels are shuffled among the season's players while their playing time stays put:
    the control that severs the link between a player's traits and his label.
    """
    anc = res.anc
    if rng is not None:
        pids = sorted(anc)
        shuffled = rng.permutation(len(pids))
        anc = {p: res.anc[pids[j]] for p, j in zip(pids, shuffled)}
    out = np.zeros((len(res.minutes), 3))
    for t, mins in enumerate(res.minutes):
        tot = sum(mins.values())
        out[t] = sum(sec * anc[p] for p, sec in mins.items()) / tot
    return out


def style_dataset(worlds, burn_in: int = 5, control_rng: np.random.Generator | None = None, within_season: bool = True):
    """Stack (ancestry, style) over every team-season of the given worlds.

    With `within_season`, each season's team values are centred on that season's league mean, so only differences
    *between teams in the same season* are explained. Otherwise league-wide season-to-season movement (e.g. a season
    with more Goliaths everywhere is slower everywhere) leaks into the regression, and into its permutation control.
    """
    X, Y = [], []
    for w in worlds:
        for res in w.league.history[burn_in:]:
            x, y = team_ancestry(res, control_rng), team_style(res)
            if within_season:
                x, y = x - x.mean(0), y - y.mean(0)
            X.append(x)
            Y.append(y)
    return np.vstack(X), np.vstack(Y)


def ols(X: np.ndarray, y: np.ndarray):
    """Regress y on [1, goliath share, elf share]. Returns (coefficients, R^2)."""
    A = np.column_stack([np.ones(len(X)), X[:, 1], X[:, 2]])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res, ss_tot = np.sum((y - A @ beta) ** 2), np.sum((y - y.mean()) ** 2)
    return beta, float(1.0 - ss_res / ss_tot)


def style_report(X: np.ndarray, Y: np.ndarray) -> dict:
    return {name: dict(zip(("coef_goliath", "coef_elf", "r2"), (*ols(X, Y[:, i])[0][1:], ols(X, Y[:, i])[1])))
            for i, name in enumerate(STYLE)}


# ---- parallel workers -------------------------------------------------------------------------------------------
class _SeasonLite:
    """The few SeasonResult fields the style analysis needs (cheap to send between processes)."""

    def __init__(self, t):
        self.team_box, self.opp_box, self.team_poss, self.team_games, self.minutes, self.anc = t


class WorldLite:
    def __init__(self, rows):
        self.league = type("League", (), {"history": [_SeasonLite(t) for t in rows]})()


def run_style_world(args):
    """(seed, years, overrides) -> per-season style inputs plus the mean pro race mix."""
    from dynasty_sim.history.sweep import _apply
    seed, years, overrides = args
    P = load_params()
    _apply(P, overrides)
    w = World(seed, P, history=False).run(years)
    rows = [(r.team_box, r.opp_box, r.team_poss, r.team_games, r.minutes, r.anc) for r in w.league.history]
    return rows, np.mean([r.race_share for r in w.records[5:]], 0)


def run_worlds_parallel(seeds, years: int, overrides=None, workers: int = 4):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_style_world, [(s, years, overrides) for s in seeds]))


def inside_table(metrics: dict, table: dict) -> tuple[bool, list]:
    """Whether every metric we can compute sits inside its validation range; also which ones do not."""
    bad = [k for k, (lo, hi) in table.items() if k in metrics
           and ((lo is not None and metrics[k] < lo) or metrics[k] > hi)]
    return not bad, bad


def run_stability_seed(args) -> dict:
    """One 100-year world reduced to what acceptance test (b) compares."""
    from dynasty_sim.engine.calibrate import validation_table
    seed, years, overrides = args
    from dynasty_sim.history.sweep import _apply
    P = load_params()
    _apply(P, overrides)
    w = World(seed, P, history=False).run(years)
    rec = w.records
    win = lambda lo, hi: [r for r in rec if lo <= r.year <= hi]
    mean_z = lambda rs: np.mean([r.league_mean_z for r in rs], 0)
    def distinct(rs):                     # every distinct pro seen in the window, once (own-race z)
        ids = np.concatenate([r.league_ids for r in rs])
        z = np.vstack([r.league_z for r in rs])
        return z[np.unique(ids, return_index=True)[1]]
    pooled = lambda rs: np.quantile(distinct(rs), 0.99, axis=0)
    pop_p99 = lambda rs: np.mean([r.pop_p99_z for r in rs], 0)
    table = validation_table()
    seasons = []
    for r in rec[5:]:
        m = dict(r.metrics)
        m["best_mov"] = r.top_mov
        ok, bad = inside_table(m, table)
        seasons.append({"year": r.year, "human_share": float(r.race_share[0]), "inside": ok, "bad": bad})
    return {"seed": seed,
            "mean_early": mean_z(win(10, 20)), "mean_late": mean_z(win(years - 9, years)),     # window against window: one season is too noisy
            "p99_early": pooled(win(10, 20)), "p99_late": pooled(win(years - 9, years)),
            "z_early": distinct(win(10, 20)), "z_late": distinct(win(years - 9, years)),
            "pop_p99_early": pop_p99(win(10, 20)), "pop_p99_late": pop_p99(win(years - 9, years)),
            "seasons": seasons, "pop_by_race": rec[-1].pop_by_race / P.K, "houses_alive": rec[-1].houses_alive}


def run_stability(seeds, years: int = 100, overrides=None, workers: int = 4) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_stability_seed, [(s, years, overrides) for s in seeds]))
