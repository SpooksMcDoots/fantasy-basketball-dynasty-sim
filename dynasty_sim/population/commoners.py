"""Commoners: unrelated pure-race founders drawn from fixed race distributions (the gene-flow anchor)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.types import Person
from dynasty_sim.genetics.genome import make_founder


def draw_death_year(anc: np.ndarray, birth_year: int, age: int, rng: np.random.Generator, P: Params) -> int:
    """Lifespan ~ N(mu, sd) conditioned on being alive at `age` (falls back to age+2)."""
    mu, sd = float(anc @ P.lifespan_mu), float(anc @ P.lifespan_sd)
    for _ in range(25):
        life = rng.normal(mu, sd)
        if life > age + 1:
            return birth_year + int(round(life))
    return birth_year + age + 2


def spawn_adult(pop, race: int, year: int, d_rng: np.random.Generator, g_rng: np.random.Generator,
                sex: Optional[str] = None, house_id: Optional[int] = None, genome=None) -> Person:
    """Register a new unrelated adult founder in the population."""
    P = pop.P
    adult = int(P.adult_age[race])
    span = max(1, int(0.35 * (P.lifespan_mu[race] - adult)))
    age = adult + int(d_rng.integers(0, span))
    sex = sex or ("F" if d_rng.random() < 0.5 else "M")
    pid = pop.ids.next()
    pop.ped.add(pid)
    genome = genome if genome is not None else make_founder(race, g_rng, P)
    p = Person(id=pid, sex=sex, birth_year=year - age, genome=genome, house_id=house_id,
               death_year=draw_death_year(genome.ancestry, year - age, age, d_rng, P))
    pop.people[pid] = p
    return p


def immigrate(pop, year: int, d_rng: np.random.Generator, g_rng: np.random.Generator) -> int:
    """Each race receives commoner adults in proportion to its shortfall from K."""
    P, n_new = pop.P, 0
    counts = pop.race_counts()
    for race in range(len(counts)):
        deficit = P.K[race] - counts[race]
        for _ in range(max(0, int(round(P.demo["immigration_rate"] * deficit)))):
            spawn_adult(pop, race, year, d_rng, g_rng)
            n_new += 1
    return n_new
