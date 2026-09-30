from __future__ import annotations

from typing import Optional

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.types import N_LOCI, Genome, Person
from dynasty_sim.genetics.hybrid import (
    hybrid_extra_variance_z, is_f1, is_pure, lineage_distance,
    pair_distance, pair_heterozygosity,
)
from dynasty_sim.genetics.pedigree import Pedigree


def make_child_genome(m: Genome, f: Genome, rng: np.random.Generator, P: Params, F: float) -> Genome:
    """Infinitesimal model: midparent + Gaussian segregation noise shrunk by parental inbreeding.

    F is the child's inbreeding coefficient (from the pedigree). Extra terms:
      * Castle-Wright hybrid variance (1.0 for mixed x mixed, 0.5 for pure x mixed backcrosses).
      * F1 "developmental instability" (DESIGN OVERRIDE, see config f1_instability_coef).
    """
    anc = 0.5 * (m.ancestry + f.ancestry)
    rho = P.mean_reversion
    seg_var = P.h2 * (1.0 - rho * rho / 2.0) * (1.0 - (m.F + f.F) / 2.0)      # = h2/2 * (1-F) when rho = 1

    n_mixed = (not is_pure(m.ancestry)) + (not is_pure(f.ancestry))
    h_rec = {0: 0.0, 1: 0.5, 2: 1.0}[n_mixed]
    hyb_var = h_rec * hybrid_extra_variance_z(anc, P) if h_rec else 0.0

    f1_var = 0.0
    if is_f1(m.ancestry, f.ancestry):
        f1_var = (P.f1_instability_coef * lineage_distance(anc, P)) ** 2 * P.f1_apply

    A = rho * 0.5 * (m.A + f.A) + rng.normal(0.0, np.sqrt(seg_var + hyb_var + f1_var))
    idx = np.arange(N_LOCI)
    loci = np.stack([m.loci[idx, rng.integers(0, 2, N_LOCI)],
                     f.loci[idx, rng.integers(0, 2, N_LOCI)]], axis=1)
    hyb_gen = 0 if is_pure(anc) else 1 + max(m.hyb_gen, f.hyb_gen)
    return Genome(A=A, loci=loci, ancestry=anc, F=F,
                  E_perm=rng.normal(0.0, np.sqrt(1.0 - P.h2)), hyb_gen=hyb_gen)


def birth(mother: Person, father: Person, child_id: int, year: int,
          rng: np.random.Generator, P: Params, ped: Pedigree) -> Person:
    F = ped.add(child_id, mother.id, father.id)
    g = make_child_genome(mother.genome, father.genome, rng, P, F)
    return Person(id=child_id, sex="F" if rng.random() < 0.5 else "M", birth_year=year,
                  genome=g, mother_id=mother.id, father_id=father.id)


def conception_prob(m: Genome, f: Genome, mother_age: float, P: Params,
                    child_F: Optional[float] = None) -> float:
    """Annual conception probability for a couple.

    base rate (ancestry-blended) * fertile-age window * hybrid factor exp(-k*D*H)
    * inbreeding gameplay penalty max(0, 1 - k_F * F_child).
    """
    lo, hi = (m.ancestry @ P.fertile_female)          # blended by mother's ancestry
    if not (lo <= mother_age <= hi):
        return 0.0
    p = float(m.ancestry @ P.base_conception)
    D, H = pair_distance(m.ancestry, f.ancestry, P), pair_heterozygosity(m.ancestry, f.ancestry)
    p *= float(np.exp(-P.hybrid_fert_k * D * H))
    if child_F is not None:
        p *= max(0.0, 1.0 - P.fertility_penalty_per_F * child_F)
    return p


def excess_child_mortality(F: float, P: Params) -> float:
    return P.child_mortality_per_F * F


def viability_mortality(g: Genome, P: Params) -> float:
    """Extra pre-reproductive mortality for extreme height or strength (own-population z beyond viability_z0)."""
    from dynasty_sim.core.types import TRAIT_IDX
    z = g.A + g.E_perm
    ex = [max(0.0, abs(float(z[TRAIT_IDX[t]])) - P.viability_z0) ** 2 for t in ("height", "strength")]
    return P.viability_coef * sum(ex)
