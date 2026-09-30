"""Run many worlds in parallel and summarise their dashboards and alarms (the M6 acceptance harness)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from dynasty_sim.config import load_params
from dynasty_sim.history import alarms as A
from dynasty_sim.league.world import World


@dataclass
class SeedSummary:
    seed: int
    alarms: list = field(default_factory=list)          # "kind: detail"
    houses_alive: int = 0
    min_race_ratio: float = 0.0                          # min over races of living / K at the last year
    max_title_share: float = 0.0                         # worst 30-season single-house share (after burn-in)
    decade_top_mov: list = field(default_factory=list)
    mean_dynasty_share: float = 0.0
    max_mean_F: float = 0.0
    race_share: list = field(default_factory=list)       # mean pro race mix after burn-in
    metrics_in_range: float = 0.0                        # share of post-burn-in seasons inside the validation table (info)


def _apply(P, overrides: dict | None) -> None:
    for path, v in (overrides or {}).items():
        section, key = path.split(".", 1)
        target = {"league": P.league, "demo": P.demo, "dev": P.dev, "injury": P.injury}[section]
        if isinstance(v, dict) and isinstance(target.get(key), dict):
            target[key] = {**target[key], **v}
        else:
            target[key] = v


def run_seed(args) -> SeedSummary:
    seed, years, overrides = args
    P = load_params()
    _apply(P, overrides)
    w = World(seed, P, history=False).run(years)
    rec = w.records
    late = [r for r in rec if r.year > A.BURN_IN]
    al = A.check_alarms(rec)
    shares = [np.bincount([r.champion_house for r in late[i:i + 30]], minlength=9).max() / 30
              for i in range(0, max(1, len(late) - 29), 5)]
    return SeedSummary(
        seed=seed, alarms=[f"{a.kind}: {a.detail}" for a in al], houses_alive=rec[-1].houses_alive,
        min_race_ratio=float(np.min(rec[-1].pop_by_race / P.K)), max_title_share=float(max(shares)),
        decade_top_mov=[float(np.mean([r.top_mov for r in late[i:i + 10]])) for i in range(0, len(late) - 9, 10)],
        mean_dynasty_share=float(np.mean([r.dynasty_share for r in late])),
        max_mean_F=float(max(r.mean_F for r in rec)),
        race_share=[float(x) for x in np.mean([r.race_share for r in late], 0)],
    )


def sweep(seeds, years: int = 100, overrides: dict | None = None, workers: int = 4) -> list[SeedSummary]:
    jobs = [(s, years, overrides) for s in seeds]
    if workers <= 1:
        return [run_seed(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_seed, jobs))


def report(results: list[SeedSummary]) -> str:
    kinds: dict[str, int] = {}
    for r in results:
        for k in {a.split(":")[0] for a in r.alarms}:
            kinds[k] = kinds.get(k, 0) + 1
    clean = sum(1 for r in results if not r.alarms)
    rs = np.mean([r.race_share for r in results], 0)
    return (f"{len(results)} seeds: {clean} with no alarms; seeds alarming by kind {kinds}; "
            f"houses alive min {min(r.houses_alive for r in results)}; min race/K {min(r.min_race_ratio for r in results):.2f}; "
            f"worst 30y house share {max(r.max_title_share for r in results):.2f}; "
            f"worst decade top MOV {max((max(r.decade_top_mov) for r in results if r.decade_top_mov), default=float('nan')):.1f}; "
            f"dynasty share {np.mean([r.mean_dynasty_share for r in results]):.2f}; max mean F {max(r.max_mean_F for r in results):.3f}; "
            f"pro race mix H/G/E {np.round(rs, 2)}")
