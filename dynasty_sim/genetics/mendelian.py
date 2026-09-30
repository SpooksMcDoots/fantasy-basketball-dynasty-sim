"""Rare, legible congenital traits. Fictional; effects in phenotypic-SD units."""
from __future__ import annotations

import numpy as np

from dynasty_sim.core.types import LOCUS_IDX, N_TRAITS, TRAIT_IDX

# locus -> (mode, {trait: SD effect}); recessive = needs 2 variant copies,
# dominant = 1 or 2 copies, additive = per copy.
EFFECTS = {
    "titan":      ("recessive", {"height": 2.0}),
    "hawkeye":    ("dominant",  {"vision": 1.0}),
    "brittle":    ("recessive", {"injury_res": -2.0}),
    "longwind":   ("additive",  {"endurance": 0.5}),
    "stonehands": ("recessive", {"touch": -1.5}),
    "late_bloom": ("recessive", {}),
}
TITAN_INJURY_HAZARD_MULT = 3.0
LATE_BLOOM_WINDOW_YEARS = 3.0


def mendelian_effects(loci: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (z-shift vector [N_TRAITS], side effects for injury/learning systems)."""
    copies = loci.sum(axis=1)
    dz = np.zeros(N_TRAITS)
    for name, (mode, eff) in EFFECTS.items():
        c = int(copies[LOCUS_IDX[name]])
        w = {"recessive": float(c == 2), "dominant": float(c >= 1), "additive": float(c)}[mode]
        for trait, sd in eff.items():
            dz[TRAIT_IDX[trait]] += w * sd
    extras = {
        "injury_hazard_mult": TITAN_INJURY_HAZARD_MULT if copies[LOCUS_IDX["titan"]] == 2 else 1.0,
        "learning_window_extra_years": LATE_BLOOM_WINDOW_YEARS if copies[LOCUS_IDX["late_bloom"]] == 2 else 0.0,
    }
    return dz, extras
