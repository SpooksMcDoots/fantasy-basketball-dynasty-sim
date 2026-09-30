"""Injury hazard (driven by workload, resilience and size) and severity (scaled by age)."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params

TYPES = ("ankle", "knee", "other")


def hazard(minutes: np.ndarray, hazard_mult: np.ndarray, P: Params) -> np.ndarray:
    """Probability of an injury this game, per rostered available player."""
    i = P.injury
    return np.minimum(0.5, (i["hazard_game"] + i["hazard_min"] * minutes) * hazard_mult)


def draw_injuries(p: np.ndarray, ae: np.ndarray, rng: np.random.Generator, P: Params):
    """Given per-player injury probabilities, return (hit_mask, type_idx, games_missed).

    games_missed is 0 for injuries the player plays through (1 - missed_frac of them).
    """
    i = P.injury
    hit = rng.random(len(p)) < p
    n = int(hit.sum())
    types = rng.choice(len(TYPES), size=n, p=np.array([i["type_weights"][t] for t in TYPES]))
    costs = rng.random(n) < i["missed_frac"]
    sev = rng.lognormal(np.log(i["sev_median"]), i["sev_sigma"], size=n)
    sev = sev * np.where(types == 1, i["knee_mult"], 1.0) * np.maximum(0.7, 1.0 + i["age_sev_slope"] * (ae[hit] - 25.0))
    games = np.where(costs, np.maximum(1, np.round(sev)), 0).astype(int)
    return hit, types, games
