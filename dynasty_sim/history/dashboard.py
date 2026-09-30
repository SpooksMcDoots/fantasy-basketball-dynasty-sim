"""Per-year snapshots of the world, all measured against the frozen year-0 reference (never the current league)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynasty_sim.core.types import RACES, TRAITS
from dynasty_sim.engine import schema as S
from dynasty_sim.genetics.genome import adult_phenotype_z, to_units


@dataclass
class YearRecord:
    year: int
    league_mean_z: np.ndarray            # [12] mean of pros' current traits, in own-race frozen-table SD units
    league_p99_z: np.ndarray             # [12]
    pop_mean_z: np.ndarray               # [12] living adults' developed phenotype
    pop_p99_z: np.ndarray
    mean_F: float                        # mean inbreeding coefficient of the living population
    top_mov: float                       # best team's margin of victory
    win_sd: float                        # SD of team win% across the league
    champion_team: int
    champion_house: int
    race_share: np.ndarray               # [3] share of pro roster slots by dominant ancestry
    dynasty_share: float                 # share of pro slots held by house-bred/house-affiliated people
    metrics: dict = field(default_factory=dict)   # league game metrics (pace, ortg, efg, ...)
    pop_by_race: np.ndarray = None       # [3] living population by race
    league_z: np.ndarray = None          # [n_pros, 12] every pro's own-race z this year (for pooled window percentiles)
    league_ids: np.ndarray = None        # [n_pros] person ids, so a window can count each player once
    houses_alive: int = 0


def season_metrics(res) -> dict:
    """Game-level league metrics from a SeasonResult's box sums."""
    T, poss, n = res.totals, res.poss, max(res.n_games, 1)
    fga, fta, fgm, tpm, tpa, ftm = T[S.ST_FGA], T[S.ST_FTA], T[S.ST_FGM], T[S.ST_TPM], T[S.ST_TPA], T[S.ST_FTM]
    elapsed = res.seconds / n / 10.0                                  # minutes per team-game per player slot -> seconds
    return {
        "pace": poss / n / 2 * 2880.0 / max(elapsed, 1e-9),
        "ortg": 100 * T[S.ST_PTS] / poss,
        "ppg": T[S.ST_PTS] / n / 2,
        "efg": (fgm + 0.5 * tpm) / fga,
        "ts": T[S.ST_PTS] / (2 * (fga + 0.44 * fta)),
        "tov_pct": 100 * T[S.ST_TOV] / (fga + 0.44 * fta + T[S.ST_TOV]),
        "orb_pct": 100 * T[S.ST_ORB] / (T[S.ST_ORB] + T[S.ST_DRB]),
        "fta_fga": fta / fga, "tpa_fga": tpa / fga, "tp_pct": tpm / tpa, "ft_pct": ftm / fta,
        "ast": T[S.ST_AST] / n / 2, "stl": T[S.ST_STL] / n / 2, "blk": T[S.ST_BLK] / n / 2, "pf": T[S.ST_PF] / n / 2,
        "home_margin": res.home_margin_sum / n, "home_win": res.home_wins / n,
    }


def _z(units: np.ndarray, ref) -> np.ndarray:
    return (units - ref.mu) / ref.sd


def snapshot(world, res) -> YearRecord:
    lg, pop, P, ref = world.league, world.pop, world.P, world.league.ref
    # z-scores are against each person's own race table (frozen config), so a shifting race mix among the pros
    # cannot masquerade as genetic drift; race composition is tracked separately in race_share.
    sts = [lg.states[p] for t in lg.teams for p in t.roster]
    zp = np.array([(s.units() - s.mu) / s.sd for s in sts])
    adults = [p for p in pop.alive_people() if world.year - p.birth_year >= P.adult_age[p.race]]
    zpop = np.array([adult_phenotype_z(p.genome, P) for p in adults])
    slots = [p for t in lg.teams for p in t.roster]
    races = np.array([lg.states[p].race for p in slots])
    dyn = sum(1 for p in slots if lg.persons[p].house_id is not None and p in pop.people)
    alive = pop.alive_people()
    return YearRecord(
        year=world.year,
        league_mean_z=zp.mean(0), league_p99_z=np.quantile(zp, 0.99, axis=0),
        pop_mean_z=zpop.mean(0), pop_p99_z=np.quantile(zpop, 0.99, axis=0),
        mean_F=float(np.mean([p.genome.F for p in alive])),
        top_mov=float(res.mov.max()), win_sd=float(res.wins.std() / P.league["games"]),
        champion_team=res.champion, champion_house=lg.teams[res.champion].house_id,
        race_share=np.bincount(races, minlength=len(RACES)) / len(races),
        dynasty_share=dyn / len(slots), metrics=season_metrics(res),
        pop_by_race=pop.race_counts(), houses_alive=len(pop.living_houses()), league_z=zp,
        league_ids=np.array(slots),
    )


def gini(counts) -> float:
    x = np.sort(np.asarray(counts, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def decade_table(records: list[YearRecord]) -> list[dict]:
    """One row per decade: means of the headline series (used for printing / export)."""
    rows = []
    for d in range(0, len(records), 10):
        chunk = records[d:d + 10]
        rows.append({
            "years": f"{chunk[0].year}-{chunk[-1].year}",
            "league_mean_z": np.mean([r.league_mean_z for r in chunk], 0),
            "league_p99_z": np.mean([r.league_p99_z for r in chunk], 0),
            "pop_mean_z": np.mean([r.pop_mean_z for r in chunk], 0),
            "mean_F": float(np.mean([r.mean_F for r in chunk])),
            "top_mov": float(np.mean([r.top_mov for r in chunk])),
            "win_sd": float(np.mean([r.win_sd for r in chunk])),
            "title_gini": gini(np.bincount([r.champion_house for r in chunk], minlength=9)[1:]),
            "race_share": np.mean([r.race_share for r in chunk], 0),
            "dynasty_share": float(np.mean([r.dynasty_share for r in chunk])),
        })
    return rows
