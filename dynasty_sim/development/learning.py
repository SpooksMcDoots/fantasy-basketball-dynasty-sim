"""Skill growth toward a genetic ceiling, scaled by learning rate and the race's critical learning window."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params

SKILLS = ("touch", "vision", "temperament")     # order of PlayerState.skill_z
DECLINING = (True, True, False)                 # temperament does not erode with age


def window_mult(age: float, race: int, P: Params) -> float:
    lo, hi, mult = P.learn_window[race]
    return float(mult) if lo <= age <= hi else 1.0


def initial_gap(ae: float, P: Params) -> float:
    """How far below the genetic ceiling a player of effective age `ae` starts (SD)."""
    return P.dev["skill_start_gap"] * P.dev["skill_gap_decay"] ** max(0.0, ae - 20.0)


def skill_step(skill: np.ndarray, target: np.ndarray, lr: float, age: float, ae: float, race: int,
               rng: np.random.Generator, P: Params) -> np.ndarray:
    d = P.dev
    k = d["skill_k"] * lr * window_mult(age, race, P)
    out = skill + k * (target - skill) + rng.normal(0.0, d["skill_noise"], size=skill.shape)
    if ae >= d["skill_decline_age_eff"]:
        out = out - d["skill_decline"] * np.array(DECLINING, float)
    return np.minimum(out, target + 0.5)
