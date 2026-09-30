"""Ancestry helpers: distance, heterozygosity, hybrid variance, heterosis."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.types import N_TRAITS, TRAIT_IDX

PURE_TOL = 0.999
_HETEROSIS_TRAITS = (TRAIT_IDX["endurance"], TRAIT_IDX["injury_res"])


def is_pure(anc: np.ndarray) -> bool:
    return float(anc.max()) >= PURE_TOL


def is_f1(anc_m: np.ndarray, anc_f: np.ndarray) -> bool:
    return is_pure(anc_m) and is_pure(anc_f) and int(anc_m.argmax()) != int(anc_f.argmax())


def pair_distance(anc_m: np.ndarray, anc_f: np.ndarray, P: Params) -> float:
    """Expected lineage distance between a random allele from each parent."""
    return float(anc_m @ P.D @ anc_f)


def pair_heterozygosity(anc_m: np.ndarray, anc_f: np.ndarray) -> float:
    """Probability the two parental alleles come from different races."""
    return float(1.0 - anc_m @ anc_f)


def own_heterozygosity(anc: np.ndarray) -> float:
    return float(1.0 - anc @ anc)


def lineage_distance(anc: np.ndarray, P: Params) -> float:
    """Mixing-weighted mean D over the race pairs an individual descends from (0 if pure)."""
    h = own_heterozygosity(anc)
    if h < 1e-9:
        return 0.0
    return float(anc @ P.D @ anc / h)


def hybrid_extra_variance_z(anc: np.ndarray, P: Params) -> np.ndarray:
    """Castle-Wright extra segregation variance for mixed x mixed matings, in child z units.

    sum_{r<s} 4*a_r*a_s*(mu_r-mu_s)^2 / (8*n_eff), computed in native units then divided
    by the child's ancestry-weighted SD^2.

    The factor is 4*a_r*a_s, not the spec's 2*a_r*a_s: at 50/50 ancestry it must reduce to the textbook F2
    segregation variance (delta mu)^2 / (8 n), and the spec's form gives exactly half of that.
    """
    dmu2 = (P.race_mu[:, None, :] - P.race_mu[None, :, :]) ** 2   # [R,R,T]
    w = np.triu(4.0 * np.outer(anc, anc), k=1)                     # [R,R]
    native = np.einsum("rs,rst->t", w, dmu2) / (8.0 * P.n_eff)
    return native / (anc @ P.race_sd) ** 2


def heterosis_z(anc: np.ndarray, hyb_gen: int, P: Params) -> np.ndarray:
    """Heterosis (gen>=1) and outbreeding depression (gen>=2) on endurance and injury resilience."""
    dz = np.zeros(N_TRAITS)
    if hyb_gen < 1:
        return dz
    D = lineage_distance(anc, P)
    val = P.heterosis_coef * D * 0.5 ** (hyb_gen - 1)
    if hyb_gen >= 2:
        val -= P.outbreeding_dep_coef * D * own_heterozygosity(anc)
    dz[list(_HETEROSIS_TRAITS)] = val
    return dz
