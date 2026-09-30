import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference, reference_to_json
from dynasty_sim.core.rng import RngTree
from dynasty_sim.core.types import N_TRAITS, RACE_IDX, TRAIT_IDX, Person
from dynasty_sim.genetics.genome import adult_phenotype_z, make_founder
from dynasty_sim.genetics.inheritance import make_child_genome, viability_mortality
from dynasty_sim.history import alarms as A
from dynasty_sim.history.dashboard import YearRecord, decade_table, gini
from dynasty_sim.league import draft as D
from dynasty_sim.league.season import League
from dynasty_sim.league.world import World
from dynasty_sim.population.demography import Population

P, REF = load_params(), frozen_reference()
HUMAN = RACE_IDX["human"]


# ---- stability mechanisms (spec 4.2 counters) ---------------------------------------------------
def test_tradeoff_tall_extremes_lose_lateral_speed():
    rng = np.random.default_rng(0)
    g = make_founder(HUMAN, rng, P)
    g.A[:], g.E_perm[:], g.loci[:] = 0.0, 0.0, 0
    base = adult_phenotype_z(g, P)[TRAIT_IDX["agility"]]
    g.A[TRAIT_IDX["height"]] = 1.0                      # below the 1.5 SD threshold: no penalty
    assert adult_phenotype_z(g, P)[TRAIT_IDX["agility"]] == pytest.approx(base)
    g.A[TRAIT_IDX["height"]] = 3.5                      # +2.0 SD beyond the threshold -> 0.6 SD slower
    assert adult_phenotype_z(g, P)[TRAIT_IDX["agility"]] == pytest.approx(base + 0.6)


def test_viability_selection_only_bites_beyond_three_sd():
    rng = np.random.default_rng(1)
    g = make_founder(HUMAN, rng, P)
    g.A[:], g.E_perm[:] = 0.0, 0.0
    assert viability_mortality(g, P) == 0.0
    g.A[TRAIT_IDX["height"]] = 3.0
    assert viability_mortality(g, P) == 0.0
    g.A[TRAIT_IDX["height"]] = 5.0                      # 2 SD past: 0.02 * 4
    assert viability_mortality(g, P) == pytest.approx(0.08)
    g.A[TRAIT_IDX["strength"]] = -4.0                   # negative extremes count too: + 0.02 * 1
    assert viability_mortality(g, P) == pytest.approx(0.10)


def test_mean_reversion_pulls_extreme_parents_toward_the_mean_but_keeps_variance():
    rng = np.random.default_rng(2)
    n = 20_000
    tall = [make_founder(HUMAN, rng, P) for _ in range(2)]
    for g in tall:
        g.A[:] = 2.0
    kids = np.array([make_child_genome(tall[0], tall[1], rng, P, 0.0).A for _ in range(4000)])
    assert np.allclose(kids.mean(0), P.mean_reversion * 2.0, atol=0.06)          # 0.95 * midparent
    a = np.array([make_founder(HUMAN, rng, P).A for _ in range(n)])
    pairs = [make_child_genome(make_founder(HUMAN, rng, P), make_founder(HUMAN, rng, P), rng, P, 0.0).A for _ in range(6000)]
    assert np.allclose(np.var(pairs, 0), P.h2, rtol=0.08)                         # V_A stays h2 across generations


def test_mate_choice_prefers_athletic_candidates_only_when_lambda_positive():
    from dynasty_sim.population.demography import pick_by_score
    rng = np.random.default_rng(3)
    scores = np.array([0.0, 0.0, 2.0])
    for lam in (0.0, 0.6, 1.5):
        expected = np.exp(2 * lam) / (2 + np.exp(2 * lam))                 # softmax weight of the score-2 candidate
        share = np.mean([pick_by_score(scores, lam, rng) == 2 for _ in range(8000)])
        assert share == pytest.approx(expected, abs=0.02), (lam, share, expected)


def test_dynasty_pipeline_runs_and_children_inherit_selected_parents():
    pop = Population(P, RngTree(4))
    pop.seed()
    for _ in range(25):
        pop.step()
    kids = [p for p in pop.people.values() if p.mother_id is not None and p.house_id is not None]
    founders = [p for p in pop.people.values() if p.mother_id is None and p.house_id is not None]
    assert len(kids) > 20 and len(founders) == P.demo["house_founders"] * 8
    from dynasty_sim.population.demography import _athletic_score
    # house lineages start from the pro-caliber tail, so their children beat an unselected baseline on average
    base = np.mean([_athletic_score(Person(0, "M", 0, make_founder(HUMAN, np.random.default_rng(i), P))) for i in range(300)])
    assert np.mean([_athletic_score(k) for k in kids]) > base


# ---- contracts, entrants ------------------------------------------------------------------------
def _league(seed=1):
    pop = Population(load_params(), RngTree(seed))
    pop.seed()
    lg = League(pop)
    lg.found()
    lg.year = 5
    return lg


def test_expiring_contracts_retain_limited_outsiders_and_family():
    lg = _league()
    team = lg.teams[0]
    for st in lg.states.values():
        st.contract_end = 10**9                          # isolate team 0
    for pid in team.roster:
        lg.states[pid].contract_end = lg.year - 1
    fam = [p for p in team.roster if lg.persons[p].house_id == team.house_id]
    outsiders = [p for p in team.roster if p not in fam]
    lg.pool = []
    n = D.expire_contracts(lg)
    L = lg.P.league
    assert len([p for p in team.roster if p in fam]) == min(len(fam), L["family_retain"])
    assert len([p for p in team.roster if p in outsiders]) == min(len(outsiders), L["retain_limit"])
    assert n == 13 - len(team.roster) and len(lg.pool) == n
    assert all(lg.states[p].team_id is None for p in lg.pool)
    assert all(lg.states[p].contract_end >= lg.year for p in team.roster)


def test_population_entrants_are_pro_caliber_and_of_draft_age():
    lg = _league(3)
    for _ in range(30):
        lg.pop.step()
    lg.year = lg.pop.year
    lg.pool = []
    D.population_entrants(lg)
    for pid in lg.pool:
        p = lg.persons[pid]
        assert D.is_pro_caliber(lg, p) and lg.age(pid) >= lg.P.adult_age[p.race]


# ---- dashboards and alarms -----------------------------------------------------------------------------
def _rec(year, mean=0.0, p99=0.0, mov=8.0, house=1, n=12):
    z = np.full(n, mean)
    pros = np.tile(np.linspace(mean, p99 + mean, 100)[:, None], (1, n))        # 99th percentile of this is ~p99+mean
    return YearRecord(year, z.copy(), np.full(n, p99), z.copy(), z.copy(), 0.0, mov, 0.15, 0, house,
                      np.array([0.6, 0.2, 0.2]), 0.3, league_z=pros,
                      league_ids=np.arange(100) + 100 * year)


def test_gini():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 8]) == pytest.approx(0.75)
    assert gini([0, 0]) == 0.0


def test_no_alarms_for_a_stationary_league():
    rng = np.random.default_rng(0)
    recs = [_rec(y, 0.0, 1.5, 8.0, int(rng.integers(1, 9))) for y in range(1, 101)]
    assert A.check_alarms(recs) == []


def test_trait_drift_alarm_fires_on_half_sd_shift_over_fifty_years():
    recs = [_rec(y, mean=0.0 if y <= 40 else 0.7, p99=1.0, house=1 + y % 8) for y in range(1, 101)]
    kinds = [a.kind for a in A.check_alarms(recs)]
    assert "trait_drift" in kinds
    small = [_rec(y, mean=0.0 if y <= 40 else 0.3, p99=1.0, house=1 + y % 8) for y in range(1, 101)]
    assert not [a for a in A.check_alarms(small) if a.kind == "trait_drift"]


def test_population_p99_drift_alarms_but_league_p99_noise_does_not():
    rng = np.random.default_rng(0)
    noisy = []
    for y in range(1, 101):
        r = _rec(y, house=1 + y % 8)
        r.league_p99_z = r.league_p99_z + rng.normal(0, 1.5, size=12)             # league p99 is deliberately not alarmed
        noisy.append(r)
    assert not [a for a in A.check_alarms(noisy) if a.kind == "trait_drift"]
    shifted = []
    for y in range(1, 101):
        r = _rec(y, house=1 + y % 8)
        r.pop_p99_z = np.full(12, 1.0 if y <= 40 else 2.0)
        shifted.append(r)
    assert any("pop_p99_z" in a.detail for a in A.check_alarms(shifted))


def test_drift_ignores_the_founding_transient():
    recs = [_rec(y, mean=5.0 if y <= A.BURN_IN else 0.0, house=1 + y % 8) for y in range(1, 101)]
    assert not A.check_alarms(recs)


def test_top_margin_and_house_title_alarms():
    hot = [_rec(y, mov=17.0, house=1 + y % 8) for y in range(1, 61)]
    assert any(a.kind == "top_mov" for a in A.check_alarms(hot))
    dyn = [_rec(y, house=3 if y % 2 else 1 + y % 8) for y in range(1, 101)]
    assert any(a.kind == "house_titles" for a in A.check_alarms(dyn))


def test_decade_table_shapes():
    rows = decade_table([_rec(y, house=1 + y % 8) for y in range(1, 41)])
    assert len(rows) == 4 and rows[0]["years"] == "1-10" and rows[0]["league_mean_z"].shape == (12,)


# ---- world and reference ------------------------------------------------------------------------------------
def test_world_records_are_complete_and_z_scored_against_reference():
    w = World(2).run(3)
    assert len(w.records) == 3
    r = w.records[-1]
    assert r.league_mean_z.shape == (N_TRAITS,) and np.isfinite(r.league_mean_z).all()
    assert np.isclose(r.race_share.sum(), 1.0) and 0.0 <= r.dynasty_share <= 1.0 and r.houses_alive >= 1
    assert 90 < r.metrics["pace"] < 115 and r.metrics["ortg"] > 80


def test_reference_serialisation_roundtrip():
    j = reference_to_json(REF)
    assert j["fingerprint"] == REF.fingerprint and len(j["mu"]) == N_TRAITS and len(j["draft_cuts"]) == 3
