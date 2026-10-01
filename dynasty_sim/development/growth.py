"""PlayerState: the yearly-evolving, native-unit trait vector of a professional player."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.types import Person, TRAIT_IDX
from dynasty_sim.development import aging, learning
from dynasty_sim.genetics.genome import adult_phenotype_z, to_units
from dynasty_sim.genetics.mendelian import mendelian_effects

_ATH_IDX = [TRAIT_IDX[t] for t in aging.ATHLETIC]
_ATH_SIGN = np.array([1.0, -1.0, 1.0, 1.0])            # agility is seconds: better = lower
_SKILL_IDX = [TRAIT_IDX[t] for t in learning.SKILLS]


@dataclass(slots=True)
class PlayerState:
    pid: int
    race: int
    mu: np.ndarray               # [12] ancestry-blended mean / SD in native units
    sd: np.ndarray
    base_units: np.ndarray       # [12] fully developed (adult) trait values from the genome
    ath_dev: np.ndarray          # [4] SD offset from base (negative while still growing)
    skill_z: np.ndarray          # [3] current skill z
    skill_target: np.ndarray     # [3] genetic ceiling z
    hazard_mult: float           # trait-driven injury hazard multiplier (excl. Mendelian)
    team_id: Optional[int] = None
    injury_type: str = ""
    games_left: int = 0
    seasons: int = 0
    low_years: int = 0
    contract_end: int = 10**9     # last year of the current contract
    log: list = field(default_factory=list)      # (year, age, ovr, games, minutes)

    def units(self) -> np.ndarray:
        u = self.base_units.copy()
        u[_ATH_IDX] += self.sd[_ATH_IDX] * self.ath_dev * _ATH_SIGN
        u[_SKILL_IDX] = self.mu[_SKILL_IDX] + self.sd[_SKILL_IDX] * self.skill_z
        return u

    def potential_units(self) -> np.ndarray:
        """Fully developed values (used to project prospects)."""
        u = self.base_units.copy()
        u[_SKILL_IDX] = self.mu[_SKILL_IDX] + self.sd[_SKILL_IDX] * self.skill_target
        return u

    @property
    def injured(self) -> bool:
        return self.games_left > 0


def new_state(person: Person, age: int, P: Params, rng: np.random.Generator, ref) -> PlayerState:
    """Create the development state of `person` at calendar age `age`, expected-value growth still ahead."""
    g = person.genome
    race = person.race
    z = adult_phenotype_z(g, P)
    units = to_units(z, g.ancestry, P)
    mu, sd = g.ancestry @ P.race_mu, g.ancestry @ P.race_sd
    ae = aging.age_eff(age, race, P)
    ath = np.array([-aging.remaining_growth(t, ae) for t in aging.ATHLETIC])
    ath = ath + rng.normal(0.0, P.dev["ath_noise"], size=4)
    target = z[_SKILL_IDX] + P.dev["skill_ceiling_z"]
    target[1:] += learning.wisdom_bonus(age, race, P)          # vision and temperament: experience already lived
    gap = learning.initial_gap(ae, P)
    z_h = (units[TRAIT_IDX["height"]] - mu[TRAIT_IDX["height"]]) / sd[TRAIT_IDX["height"]]   # within own population
    z_res = (units[TRAIT_IDX["injury_res"]] - ref.mu[TRAIT_IDX["injury_res"]]) / ref.sd[TRAIT_IDX["injury_res"]]
    inj = P.injury
    hazard = float(np.exp(-inj["res_slope"] * z_res) * np.exp(inj["height_slope"] * max(0.0, z_h - inj["height_z0"]))
                   * mendelian_effects(g.loci)[1]["injury_hazard_mult"])
    return PlayerState(person.id, race, mu, sd, units, ath, target - gap, target, hazard)


def develop_year(st: PlayerState, age: int, P: Params, rng: np.random.Generator) -> None:
    """Advance one year of development (age is the age at the start of the year)."""
    ae = aging.age_eff(age, st.race, P)
    st.ath_dev = aging.athletic_step(st.ath_dev, ae, P.decline_mult[st.race], rng, P.dev["ath_noise"])
    gain = learning.wisdom_bonus(age + 1, st.race, P) - learning.wisdom_bonus(age, st.race, P)
    st.skill_target[1:] += gain                                 # a year of experience raises the court-sense ceiling
    lr = float(st.base_units[TRAIT_IDX["learning"]])
    st.skill_z = learning.skill_step(st.skill_z, st.skill_target, lr, age, ae, st.race, rng, P)


def state_at_age(person: Person, start_age: int, age: int, P: Params, rng: np.random.Generator, ref) -> PlayerState:
    """State at `age`, developed forward from `start_age` (used to seed veterans)."""
    st = new_state(person, start_age, P, rng, ref)
    for a in range(start_age, age):
        develop_year(st, a, P, rng)
    return st
