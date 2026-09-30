from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Array layouts. Order must match config/defaults.yaml and config/races.yaml.
TRAITS = (
    "height", "ape", "vertical", "agility", "endurance", "strength",
    "hand", "touch", "vision", "learning", "injury_res", "temperament",
)
RACES = ("human", "goliath", "elf")
LOCI = ("titan", "hawkeye", "brittle", "longwind", "stonehands", "late_bloom")

N_TRAITS = len(TRAITS)
N_RACES = len(RACES)
N_LOCI = len(LOCI)

TRAIT_IDX = {t: i for i, t in enumerate(TRAITS)}
RACE_IDX = {r: i for i, r in enumerate(RACES)}
LOCUS_IDX = {l: i for i, l in enumerate(LOCI)}


@dataclass(slots=True)
class Genome:
    A: np.ndarray          # float64[N_TRAITS], additive breeding values, phenotypic-SD units
    loci: np.ndarray       # uint8[N_LOCI, 2], allele ids (0=wild, 1=variant)
    ancestry: np.ndarray   # float64[N_RACES], sums to 1
    F: float               # inbreeding coefficient
    E_perm: np.ndarray     # float64[N_TRAITS], permanent environment draw (fixed at birth)
    hyb_gen: int = 0       # 0 = pure line, 1 = F1 of pure lines, k = 1 + max(parents) otherwise


@dataclass(slots=True)
class Person:
    id: int
    sex: str                       # "F" or "M"
    birth_year: int
    genome: Genome
    mother_id: Optional[int] = None
    father_id: Optional[int] = None
    house_id: Optional[int] = None
    name: str = ""
    death_year: Optional[int] = None      # fixed at birth / seeding; None = not yet decided
    spouse_id: Optional[int] = None
    last_birth_year: Optional[int] = None
    alive: bool = True

    @property
    def race(self) -> int:
        return int(self.genome.ancestry.argmax())


@dataclass(slots=True)
class House:
    id: int
    name: str
    race: int
    head_id: Optional[int] = None
    founded_year: int = 0
    extinct_year: Optional[int] = None
    heir_policy: str = "eldest_child"


@dataclass(slots=True)
class Event:
    year: int
    kind: str                     # succession | adoption | extinction | ...
    house_id: Optional[int]
    actors: tuple = ()
    detail: str = ""
