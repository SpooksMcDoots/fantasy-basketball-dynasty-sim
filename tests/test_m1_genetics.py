import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from dynasty_sim.config import load_params
from dynasty_sim.core.ids import IdAllocator
from dynasty_sim.core.types import N_TRAITS, RACE_IDX, TRAIT_IDX, Person
from dynasty_sim.genetics.genome import additive_phenotype_z, adult_phenotype_z, make_founder, to_units
from dynasty_sim.genetics.inheritance import (
    birth, conception_prob, excess_child_mortality, make_child_genome,
)
from dynasty_sim.genetics.pedigree import Pedigree

P = load_params()
HUMAN, GOLIATH, ELF = RACE_IDX["human"], RACE_IDX["goliath"], RACE_IDX["elf"]


# ---- pedigree F ---------------------------------------------------------------
def _ped(*rows):
    ped = Pedigree()
    for pid, m, f in rows:
        ped.add(pid, m, f)
    return ped


def test_full_sib_child_F():
    ped = _ped((1, None, None), (2, None, None), (3, 1, 2), (4, 1, 2))
    assert ped.add(5, 3, 4) == pytest.approx(0.25)


def test_half_sib_child_F():
    ped = _ped((1, None, None), (2, None, None), (4, None, None), (3, 1, 2), (5, 1, 4))
    assert ped.add(6, 3, 5) == pytest.approx(0.125)


def test_first_cousin_child_F():
    ped = _ped((1, None, None), (2, None, None), (3, 1, 2), (4, 1, 2),
               (5, None, None), (7, None, None), (6, 3, 5), (8, 4, 7))
    assert ped.add(9, 6, 8) == pytest.approx(0.0625)


def test_parent_offspring_and_unrelated():
    ped = _ped((1, None, None), (2, None, None), (3, 1, 2), (4, None, None))
    assert ped.add(5, 1, 3) == pytest.approx(0.25)
    assert ped.add(6, 3, 4) == 0.0


def test_pedigree_rejects_bad_parent_ids():
    ped = _ped((1, None, None))
    with pytest.raises(ValueError):
        ped.add(2, 1, 5)


# ---- additive model -----------------------------------------------------------
def test_offspring_on_midparent_slope_equals_h2():
    rng = np.random.default_rng(1)
    n = 20_000
    mid = np.empty((n, N_TRAITS))
    kid = np.empty((n, N_TRAITS))
    for i in range(n):
        m, f = make_founder(HUMAN, rng, P), make_founder(HUMAN, rng, P)
        c = make_child_genome(m, f, rng, P, F=0.0)
        mid[i] = 0.5 * (additive_phenotype_z(m) + additive_phenotype_z(f))
        kid[i] = additive_phenotype_z(c)
    slope = ((mid - mid.mean(0)) * (kid - kid.mean(0))).sum(0) / ((mid - mid.mean(0)) ** 2).sum(0)
    assert np.all(np.abs(slope - P.h2) < 0.05), (slope, P.h2)


@pytest.mark.slow
def test_variance_stable_over_50_random_mating_generations():
    """Model predicts Var(A) = h2 * (1 - mean F) under drift in a finite population."""
    rng = np.random.default_rng(3)
    ped, ids, N = Pedigree(), IdAllocator(), 200
    pop = []
    for k in range(N):
        pid = ids.next()
        ped.add(pid)
        pop.append(Person(pid, "F" if k % 2 == 0 else "M", 0, make_founder(HUMAN, rng, P)))
    for gen in range(1, 51):
        women = [p for p in pop if p.sex == "F"]
        men = [p for p in pop if p.sex == "M"]
        nxt = []
        for _ in range(N):
            m, f = women[rng.integers(len(women))], men[rng.integers(len(men))]
            nxt.append(birth(m, f, ids.next(), gen, rng, P, ped))
        pop = nxt
    A = np.array([p.genome.A for p in pop])
    Fbar = float(np.mean([p.genome.F for p in pop]))
    ratio = float((A.var(axis=0) / P.h2).mean())
    assert abs(ratio - (1 - Fbar)) < 0.10, (ratio, Fbar)
    assert 0.6 < ratio < 1.15


@given(st.floats(0, 0.5), st.floats(0, 0.5))
@settings(max_examples=50, deadline=None)
def test_segregation_variance_never_negative(Fm, Ff):
    rng = np.random.default_rng(0)
    m, f = make_founder(HUMAN, rng, P), make_founder(GOLIATH, rng, P)
    m.F, f.F = Fm, Ff
    c = make_child_genome(m, f, rng, P, F=0.1)
    assert np.all(np.isfinite(c.A)) and abs(c.ancestry.sum() - 1) < 1e-12


# ---- inbreeding depression ----------------------------------------------------
def _neutral_human():
    g = make_founder(HUMAN, np.random.default_rng(0), P)
    g.A[:], g.E_perm[:], g.loci[:] = 0.0, 0.0, 0
    return g


def test_first_cousin_height_loss_near_sourced_value():
    g = _neutral_human()
    h0 = to_units(adult_phenotype_z(g, P), g.ancestry, P)[TRAIT_IDX["height"]]
    g.F = 0.0625
    h1 = to_units(adult_phenotype_z(g, P), g.ancestry, P)[TRAIT_IDX["height"]]
    assert h0 - h1 == pytest.approx(1.35, abs=0.05)   # 2.7 SD/F * .0625 * 8 cm (sourced: 1.2 cm)


def test_agility_inbreeding_raises_seconds():
    g = _neutral_human()
    z0 = adult_phenotype_z(g, P)
    g.F = 0.25
    z1 = adult_phenotype_z(g, P)
    assert z1[TRAIT_IDX["agility"]] > z0[TRAIT_IDX["agility"]]
    assert z1[TRAIT_IDX["height"]] < z0[TRAIT_IDX["height"]]


# ---- ancestry, hybrids, fertility ----------------------------------------------
def test_founder_units_match_race_table():
    rng = np.random.default_rng(5)
    g = [make_founder(GOLIATH, rng, P) for _ in range(4000)]
    h = np.array([to_units(a.A + a.E_perm, a.ancestry, P)[0] for a in g])
    assert abs(h.mean() - 232) < 0.6 and abs(h.std() - 11) < 0.6


def test_hybrid_generation_and_ancestry():
    rng = np.random.default_rng(2)
    h, g = make_founder(HUMAN, rng, P), make_founder(GOLIATH, rng, P)
    f1 = make_child_genome(h, g, rng, P, 0.0)
    f2 = make_child_genome(f1, make_child_genome(h, g, rng, P, 0.0), rng, P, 0.0)
    bc = make_child_genome(h, f1, rng, P, 0.0)
    assert f1.hyb_gen == 1 and f2.hyb_gen == 2 and bc.hyb_gen == 2
    assert np.allclose(f1.ancestry, [.5, .5, 0]) and np.allclose(bc.ancestry, [.75, .25, 0])


def test_hybrid_fertility_follows_the_configured_penalty():
    rng = np.random.default_rng(2)
    h, g, e = (make_founder(r, rng, P) for r in (HUMAN, GOLIATH, ELF))
    assert conception_prob(h, g, 25, P) / conception_prob(h, h, 25, P) == pytest.approx(np.exp(-P.hybrid_fert_k * 0.55), rel=1e-6)
    assert conception_prob(e, g, 60, P) / conception_prob(e, e, 60, P) == pytest.approx(np.exp(-P.hybrid_fert_k * 0.75), rel=1e-6)
    assert conception_prob(h, h, 60, P) == 0.0           # outside fertile window


def test_inbreeding_fertility_penalty_and_mortality():
    h = make_founder(HUMAN, np.random.default_rng(2), P)
    assert conception_prob(h, h, 25, P, child_F=0.25) == pytest.approx(0.75 * conception_prob(h, h, 25, P))
    assert excess_child_mortality(0.0625, P) == pytest.approx(0.035)


def test_hybrid_variance_ordering_height():
    """F2 wider than F1 and pure (Castle-Wright)."""
    rng = np.random.default_rng(11)
    n = 3000
    H = lambda: make_founder(HUMAN, rng, P)
    G = lambda: make_founder(GOLIATH, rng, P)
    F1 = lambda: make_child_genome(H(), G(), rng, P, 0.0)
    hh = [make_child_genome(H(), H(), rng, P, 0.0) for _ in range(n)]
    f1 = [F1() for _ in range(n)]
    f2 = [make_child_genome(F1(), F1(), rng, P, 0.0) for _ in range(n)]
    sd = lambda gs: np.std([to_units(a.A + a.E_perm, a.ancestry, P)[0] for a in gs])
    assert sd(f2) > sd(f1) > sd(hh)


def test_f1_override_makes_f1_wider_than_pure_in_standardized_units():
    """Preview of acceptance (c): F1 z-SD > pure z-SD on >= 8 of 12 traits, Levene p < 0.01."""
    from scipy.stats import levene
    rng = np.random.default_rng(21)
    n = 5000
    pure = np.array([additive_phenotype_z(make_child_genome(make_founder(HUMAN, rng, P), make_founder(HUMAN, rng, P), rng, P, 0.0)) for _ in range(n)])
    f1 = np.array([additive_phenotype_z(make_child_genome(make_founder(HUMAN, rng, P), make_founder(GOLIATH, rng, P), rng, P, 0.0)) for _ in range(n)])
    assert (f1.std(0) > pure.std(0)).sum() >= 8
    assert sum(levene(f1[:, t], pure[:, t]).pvalue < 0.01 for t in range(N_TRAITS)) >= 8
