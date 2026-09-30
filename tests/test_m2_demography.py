import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.rng import RngTree
from dynasty_sim.core.types import RACE_IDX, House, Person
from dynasty_sim.genetics.genome import make_founder
from dynasty_sim.population.demography import Population
from dynasty_sim.population.succession import resolve_head

P = load_params()
HUMAN = RACE_IDX["human"]


def _run(seed, years=100, P=P):
    pop = Population(P, RngTree(seed))
    pop.seed()
    for _ in range(years):
        pop.step()
    return pop


# ---- acceptance (M2) ------------------------------------------------------------
def test_population_and_houses_stable_10_seeds():
    for seed in range(10):
        pop = _run(seed)
        assert np.all(np.abs(pop.race_counts() / P.K - 1) <= 0.25), (seed, pop.stats())
        assert len(pop.living_houses()) >= 6, (seed, pop.stats())


@pytest.mark.slow
def test_50_seed_acceptance():
    """Population within +/-25% of K per race at year 100; >=6 of 8 houses survive in >=90% of seeds."""
    pop_ok = house_ok = 0
    for seed in range(50):
        pop = _run(seed)
        pop_ok += bool(np.all(np.abs(pop.race_counts() / P.K - 1) <= 0.25))
        house_ok += len(pop.living_houses()) >= 6
    assert pop_ok >= 45 and house_ok >= 45, (pop_ok, house_ok)


def test_same_seed_same_history():
    a, b = _run(3, 40), _run(3, 40)
    assert a.stats() == b.stats() and [(e.year, e.kind, e.house_id) for e in a.events] == \
        [(e.year, e.kind, e.house_id) for e in b.events]


# ---- rules ------------------------------------------------------------------------
def test_close_kin_never_marry_under_defaults():
    pop = _run(1)
    assert max(p.genome.F for p in pop.people.values()) < 0.25


def test_inbreeding_appears_when_kin_avoidance_off():
    P2 = load_params()
    P2.demo.update(kin_avoid_prob=0.0, kin_F_threshold=9)
    assert max(p.genome.F for seed in range(3) for p in _run(seed, P=P2).people.values()) > 0


def test_dead_have_no_spouse_and_no_births():
    pop = _run(2, 60)
    for p in pop.people.values():
        if not p.alive:
            assert p.spouse_id is None
        else:
            assert p.death_year > pop.year
    for cs in pop.children.values():
        for c in cs:
            assert pop.people[c].birth_year <= pop.year


def test_heads_stay_in_their_house():
    pop = _run(4, 60)
    for h in pop.living_houses():
        if h.head_id is not None:
            head = pop.people[h.head_id]
            assert head.alive and head.house_id == h.id


def test_commoners_immigrate_as_unrelated_pure_founders():
    pop = _run(5, 30)
    commoners = [p for p in pop.people.values() if p.mother_id is None and p.house_id is None]
    assert commoners and all(p.genome.F == 0 and p.genome.hyb_gen == 0 for p in commoners)


# ---- succession ladder --------------------------------------------------------------
class _Fx:
    def __init__(self):
        self.pop = Population(P, RngTree(0))
        self.pop.year = 50
        self.rng = np.random.default_rng(0)
        self.pop.houses[1] = House(1, "House 1", HUMAN)
        self.pop.houses[2] = House(2, "House 2", HUMAN)

    def add(self, birth_year, sex="M", mother=None, father=None, house=1, alive=True):
        pid = self.pop.ids.next()
        self.pop.ped.add(pid, mother, father)
        p = Person(pid, sex, birth_year, make_founder(HUMAN, self.rng, P), mother, father, house,
                   death_year=200, alive=alive)
        self.pop.people[pid] = p
        for par in (mother, father):
            if par is not None:
                self.pop.children[par].append(pid)
        return p

    def resolve(self, head):
        head.alive = False
        resolve_head(self.pop, self.pop.houses[1], head, self.pop.year)
        return self.pop.houses[1].head_id


def test_succession_eldest_child_in_house():
    fx = _Fx()
    m, f = fx.add(0, "F"), fx.add(0, "M")
    older, younger = fx.add(20, mother=m.id, father=f.id), fx.add(25, mother=m.id, father=f.id)
    assert fx.resolve(f) == older.id


def test_succession_skips_child_who_left_house():
    fx = _Fx()
    m, f = fx.add(0, "F"), fx.add(0, "M")
    fx.add(20, "F", m.id, f.id, house=2)
    younger = fx.add(25, mother=m.id, father=f.id)
    assert fx.resolve(f) == younger.id


def test_succession_falls_back_to_sibling_then_nephew():
    fx = _Fx()
    gm, gf = fx.add(-30, "F", house=None), fx.add(-30, "M", house=None)
    head = fx.add(0, mother=gm.id, father=gf.id)
    sib = fx.add(3, mother=gm.id, father=gf.id)
    assert fx.resolve(head) == sib.id

    fx = _Fx()
    gm, gf = fx.add(-30, "F", house=None), fx.add(-30, "M", house=None)
    head = fx.add(0, mother=gm.id, father=gf.id)
    dead_sib = fx.add(3, mother=gm.id, father=gf.id, alive=False)
    nephew = fx.add(30, mother=None, father=dead_sib.id)
    assert fx.resolve(head) == nephew.id


def test_succession_nephew_cadet_and_extinction():
    fx = _Fx()
    gm, gf = fx.add(-60, "F", house=None), fx.add(-60, "M", house=None)
    head = fx.add(0, mother=gm.id, father=gf.id)
    cousin_parent = fx.add(-1, mother=gm.id, father=gf.id, house=2)      # sibling in another house
    nephew = fx.add(20, mother=None, father=cousin_parent.id, house=1)    # sibling's child, in house
    fx.add(20, house=1)                                                    # unrelated member
    assert fx.resolve(head) == nephew.id                                   # nephew rung, not "any member"

    fx = _Fx()                                                             # cadet adoption
    gm, gf = fx.add(-60, "F", house=None, alive=False), fx.add(-60, "M", house=None, alive=False)
    head = fx.add(0, mother=gm.id, father=gf.id)
    outsider = fx.add(5, mother=gm.id, father=gf.id, house=None)
    spouse = fx.add(6, "F", house=None)
    outsider.spouse_id, spouse.spouse_id = spouse.id, outsider.id
    kid = fx.add(30, mother=spouse.id, father=outsider.id, house=None)
    assert fx.resolve(head) == outsider.id
    assert spouse.house_id == 1 and kid.house_id == 1
    assert fx.pop.events[-1].kind == "adoption"

    fx = _Fx()                                                             # extinction
    head = fx.add(0)
    assert fx.resolve(head) is None
    assert fx.pop.houses[1].extinct_year == 50 and fx.pop.events[-1].kind == "extinction"
