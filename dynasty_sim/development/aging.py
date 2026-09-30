"""Athletic development and decline on effective age, shaped like ZenGM's curves at 0.1 SD per ZenGM point."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params

ATHLETIC = ("vertical", "agility", "endurance", "strength")   # order of PlayerState.ath_dev


def age_eff(age: float, race: int, P: Params) -> float:
    """Race-scaled age: 20 at the age growth completes, then one unit per (timescale) calendar years."""
    return 20.0 + (age - P.adult_age[race]) / P.timescale[race]


def increment(trait: str, ae: float) -> float:
    """Expected yearly change in SD (positive = better) for the year starting at effective age `ae`."""
    a = int(np.floor(ae))
    if a <= 21:
        return 0.20
    if a <= 25:
        return 0.10
    if trait == "vertical":
        return -0.3 if a >= 27 else 0.0
    if trait == "agility":
        return 0.0 if a < 28 else -0.2 if a <= 30 else -0.3 if a <= 35 else -0.4
    if trait == "endurance":
        return 0.0 if a < 31 else -0.2 if a <= 35 else -0.3
    # strength: generic base decline
    return 0.0 if a < 28 else -0.1 if a <= 29 else -0.2 if a <= 31 else -0.3 if a <= 34 else -0.4


def remaining_growth(trait: str, ae: float) -> float:
    """Sum of expected growth increments between `ae` and the end of growth (effective age 26)."""
    total, a = 0.0, ae
    while a < 26.0:
        total += increment(trait, a)
        a += 1.0
    return total


def athletic_step(dev: np.ndarray, ae: float, decline_mult: float, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Advance the four athletic development offsets (SD) by one year."""
    out = dev.copy()
    for i, t in enumerate(ATHLETIC):
        inc = increment(t, ae)
        if inc < 0:
            inc *= decline_mult
        out[i] += inc * rng.uniform(0.4, 1.4) + rng.normal(0.0, noise)
    return out
