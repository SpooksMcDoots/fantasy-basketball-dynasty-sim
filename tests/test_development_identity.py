"""Race development identity: Humans are quick studies who peak early, Elves climb slowly and keep gaining court sense."""
import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.types import RACE_IDX, TRAIT_IDX
from dynasty_sim.development import learning

P = load_params()
HUMAN, GOLIATH, ELF = RACE_IDX["human"], RACE_IDX["goliath"], RACE_IDX["elf"]


def _rate(race: int, age: int) -> float:
    """Fraction of the gap to the skill ceiling a typical player of the race closes in a year."""
    return P.dev["skill_k"] * P.race_mu[race, TRAIT_IDX["learning"]] * learning.window_mult(age, race, P)


def test_humans_learn_faster_than_elves_through_their_first_pro_years():
    human_first = [_rate(HUMAN, a) for a in range(int(P.adult_age[HUMAN]), int(P.adult_age[HUMAN]) + 6)]
    elf_first = [_rate(ELF, a) for a in range(int(P.adult_age[ELF]), int(P.adult_age[ELF]) + 6)]
    assert min(human_first) > 1.5 * max(elf_first)


def test_elves_take_far_longer_to_close_the_gap():
    def years_to_close(race: int, frac: float = 0.9) -> int:
        gap, age = 1.0, int(P.adult_age[race])
        while gap > 1 - frac:
            gap *= 1 - _rate(race, age)
            age += 1
        return age - int(P.adult_age[race])
    assert years_to_close(ELF) >= 2 * years_to_close(HUMAN)
    assert years_to_close(HUMAN) <= 8


def test_wisdom_accrues_with_years_lived_and_is_capped():
    cap = P.dev["wisdom_cap"]
    a = P.adult_age[ELF]
    assert learning.wisdom_bonus(a, ELF, P) == 0.0
    assert learning.wisdom_bonus(a + 10, ELF, P) == pytest.approx(10 * P.wisdom[ELF])
    assert learning.wisdom_bonus(a + 200, ELF, P) == cap
    assert learning.wisdom_bonus(a - 5, ELF, P) == 0.0
    assert P.wisdom[GOLIATH] == 0.0 < P.wisdom[HUMAN] < P.wisdom[ELF]


def test_a_veteran_elf_has_more_court_sense_headroom_than_a_veteran_human():
    from dynasty_sim.core.reference import frozen_reference
    from dynasty_sim.core.types import Person
    from dynasty_sim.development.growth import new_state
    from dynasty_sim.genetics.genome import make_founder
    rng, ref = np.random.default_rng(1), frozen_reference()
    gains = {}
    for race in (HUMAN, ELF):
        g = make_founder(race, rng, P)
        young = new_state(Person(1, "M", 0, g), int(P.adult_age[race]), P, rng, ref).skill_target
        old = new_state(Person(1, "M", 0, g), int(P.adult_age[race]) + 15, P, rng, ref).skill_target
        assert old[0] == pytest.approx(young[0])                          # shooting touch is not wisdom
        gains[race] = old[1:] - young[1:]
    assert np.all(gains[ELF] > 3 * gains[HUMAN]) and np.all(gains[HUMAN] > 0)
