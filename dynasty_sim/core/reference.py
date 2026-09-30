"""Frozen reference league used for every z-score the engine sees.

The reference is the draft-selected Human calibration league: a fixed-seed sample of the general Human
trait table, keeping the top DRAFT_FRAC on a fixed draft composite. It is built once from constants,
never from the current league, so drift is never re-normalised away.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from dynasty_sim.config import Params, load_params
from dynasty_sim.core.types import RACE_IDX, TRAIT_IDX, TRAITS

REFERENCE_SEED = 20240101
REFERENCE_N = 400_000
# Positional draft archetypes: (share of league, top fraction kept, composite weights on general-Human
# z-scores; agility is seconds so lower is better). Mixing them gives guards and bigs, not one clone.
ARCHETYPES = (
    ("guard", 0.35, 0.020, {"height": 0.15, "agility": -0.30, "touch": 0.25, "vision": 0.20, "vertical": 0.10}),
    ("wing",  0.35, 0.020, {"height": 0.50, "vertical": 0.15, "agility": -0.10, "touch": 0.15, "strength": 0.10}),
    ("big",   0.30, 0.004, {"height": 0.80, "strength": 0.10, "vertical": 0.05, "endurance": 0.05}),
)


@dataclass(frozen=True)
class Reference:
    mu: np.ndarray        # [12] trait means of the reference league (native units)
    sd: np.ndarray        # [12] trait SDs
    mr_mu: float          # standing reach + vertical (max reach at full energy)
    mr_sd: float
    rel_mu: float         # mean release height
    reach_off: float      # mean (max_reach - release): contest reach term is centred here
    wing_mu: float
    wing_sd: float
    draft_cuts: tuple     # per-archetype composite thresholds that define "selected"
    fingerprint: str      # hash of everything above, used to detect a stale calibration

    def z(self, trait: str, value):
        i = TRAIT_IDX[trait]
        return (value - self.mu[i]) / self.sd[i]


def derived_heights(units: np.ndarray):
    """units [..., 12] -> (reach, max_reach, release, wingspan) at full energy."""
    h, ape, vert = units[..., TRAIT_IDX["height"]], units[..., TRAIT_IDX["ape"]], units[..., TRAIT_IDX["vertical"]]
    reach = h * (1.33 + 0.5 * (ape - 1.0))
    return reach, reach + vert, 0.92 * reach + 0.4 * vert, h * ape


def draft_score(units: np.ndarray, P: Params, archetype: int, race: str = "human") -> np.ndarray:
    """Fixed positional draft composite on a population's own trait table (z-scores within that population)."""
    r = RACE_IDX[race]
    z = (units - P.race_mu[r]) / P.race_sd[r]
    return sum(w * z[..., TRAIT_IDX[t]] for t, w in ARCHETYPES[archetype][3].items())


def draft_z_score(z: np.ndarray, archetype: int) -> np.ndarray:
    """Draft composite from own-population z-scores (identical to draft_score, without the unit round trip)."""
    return sum(w * z[..., TRAIT_IDX[t]] for t, w in ARCHETYPES[archetype][3].items())


def select_mask(units: np.ndarray, P: Params, cuts, archetype: int, race: str = "human") -> np.ndarray:
    return draft_score(units, P, archetype, race) >= cuts[archetype]


@lru_cache(maxsize=1)
def frozen_reference() -> Reference:
    P = load_params()
    r = RACE_IDX["human"]
    rng = np.random.default_rng(REFERENCE_SEED)
    units = P.race_mu[r] + P.race_sd[r] * rng.standard_normal((REFERENCE_N, len(TRAITS)))
    cuts, parts = [], []
    for a, (_, share, frac, _) in enumerate(ARCHETYPES):
        cut = float(np.quantile(draft_score(units, P, a), 1.0 - frac))
        cuts.append(cut)
        sel = units[draft_score(units, P, a) >= cut]
        parts.append(sel[rng.choice(len(sel), int(share * 100_000), replace=True)])
    sel = np.vstack(parts)
    _, mr, rel, wing = derived_heights(sel)
    vals = [*sel.mean(0), *sel.std(0), mr.mean(), mr.std(), rel.mean(), (mr - rel).mean(),
            wing.mean(), wing.std(), *cuts]
    fp = hashlib.sha256(np.round(np.array(vals), 6).tobytes()).hexdigest()[:16]
    return Reference(sel.mean(0), sel.std(0), float(mr.mean()), float(mr.std()), float(rel.mean()),
                     float((mr - rel).mean()), float(wing.mean()), float(wing.std()), tuple(cuts), fp)


def reference_to_json(ref: Reference) -> dict:
    """Serializable form of the frozen year-0 reference (written next to every run's outputs)."""
    return {"traits": list(TRAITS), "mu": [round(float(x), 6) for x in ref.mu], "sd": [round(float(x), 6) for x in ref.sd],
            "mr_mu": ref.mr_mu, "mr_sd": ref.mr_sd, "rel_mu": ref.rel_mu, "reach_off": ref.reach_off,
            "wing_mu": ref.wing_mu, "wing_sd": ref.wing_sd, "draft_cuts": list(ref.draft_cuts),
            "draft_archetypes": [a[0] for a in ARCHETYPES], "fingerprint": ref.fingerprint,
            "seed": REFERENCE_SEED, "n": REFERENCE_N}


def legacy_cut(archetype: int, frac: float) -> float:
    """Draft-composite threshold for the top `frac` of a population (composite of independent unit-variance z's)."""
    from statistics import NormalDist
    sigma = float(np.sqrt(sum(w * w for w in ARCHETYPES[archetype][3].values())))
    return sigma * NormalDist().inv_cdf(1.0 - frac)
