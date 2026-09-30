from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.types import N_LOCI, N_RACES, N_TRAITS, TRAIT_IDX, Genome
from dynasty_sim.genetics.hybrid import heterosis_z
from dynasty_sim.genetics.mendelian import mendelian_effects


def make_founder(race: int, rng: np.random.Generator, P: Params) -> Genome:
    """Unrelated pure-race founder. A ~ N(0, h2) so total phenotypic variance is 1."""
    anc = np.zeros(N_RACES)
    anc[race] = 1.0
    alleles = (rng.random((N_LOCI, 2)) < P.locus_freq[race][:, None]).astype(np.uint8)
    return Genome(
        A=rng.normal(0.0, np.sqrt(P.h2)),
        loci=alleles,
        ancestry=anc,
        F=0.0,
        E_perm=rng.normal(0.0, np.sqrt(1.0 - P.h2)),
        hyb_gen=0,
    )


def additive_phenotype_z(g: Genome) -> np.ndarray:
    """A + E only. Used by the pure quantitative-genetics tests."""
    return g.A + g.E_perm


def adult_phenotype_z(g: Genome, P: Params) -> np.ndarray:
    """Adult phenotype in SD units relative to the individual's ancestry-blended mean."""
    z = g.A + g.E_perm + P.dep_direction * P.inbreeding_dep * g.F
    z = z + heterosis_z(g.ancestry, g.hyb_gen, P)
    z = z + mendelian_effects(g.loci)[0]
    # trade-off matrix: extreme height costs lateral speed (agility is seconds, so slower = higher)
    z_h = z[TRAIT_IDX["height"]]
    z[TRAIT_IDX["agility"]] += P.tradeoff_agility * max(0.0, z_h - P.tradeoff_z0)
    return z


def to_units(z: np.ndarray, ancestry: np.ndarray, P: Params) -> np.ndarray:
    return ancestry @ P.race_mu + (ancestry @ P.race_sd) * z
