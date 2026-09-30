"""Deterministic invented names. Draws come from an isolated narrative stream keyed by (kind, id), so naming can
never change game results and never depends on the order things are looked up. No real-world naming traditions."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from dynasty_sim.core.rng import STREAMS
from dynasty_sim.core.types import RACES

_NARRATIVE = STREAMS.index("narrative")
_GIVEN, _SURNAME, _TEAM, _HOUSE = range(4)

_POOLS = {
    "human": (["b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "c", "h", "f", "w", "j"],
              ["a", "e", "i", "o", "u", "ai", "ea"], ["n", "r", "l", "m", "s", "d", "th", "k", ""]),
    "goliath": (["gr", "kr", "br", "dr", "tor", "gor", "mor", "bor", "z", "ur", "og", "th"],
                ["a", "o", "u", "au"], ["k", "r", "n", "th", "g", "m", "d", "rk"]),
    "elf": (["l", "ael", "el", "ith", "sil", "th", "n", "v", "y", "ar", "ir", "ph"],
            ["a", "e", "i", "ae", "ia", "ei", "io"], ["n", "r", "l", "s", "th", "d", "ss", ""]),
}
_CITIES = ["Ashvale", "Brindle", "Cairnholt", "Dunmere", "Eastwick", "Fenwick", "Galloway", "Harrowgate", "Ironmoor",
           "Juniper Bay", "Kestrel Point", "Lowmarch", "Marrowdale", "Northreach", "Oakhollow", "Pinecrest",
           "Quillon", "Ravenscar", "Stonebridge", "Thornfield"]
_MASCOTS = ["Herons", "Wolves", "Anvils", "Lanterns", "Falcons", "Tides", "Wardens", "Foxes", "Bears", "Comets",
            "Harriers", "Rooks", "Stags", "Vipers", "Marlins", "Owls", "Bison", "Lynx", "Ravens", "Otters"]


class NameBook:
    def __init__(self, run_seed: int, lookup: Callable[[int], Optional[object]]):
        self.seed = int(run_seed)
        self._lookup = lookup
        self._cache: dict = {}

    def _rng(self, kind: int, ident: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(_NARRATIVE, kind, int(ident))))

    def _word(self, rng: np.random.Generator, race: str, syllables: int) -> str:
        on, vo, co = _POOLS[race]
        w = "".join(on[rng.integers(len(on))] + vo[rng.integers(len(vo))] + (co[rng.integers(len(co))] if i == syllables - 1 else "")
                    for i in range(syllables))
        return w.capitalize()

    def given(self, pid: int) -> str:
        key = ("g", pid)
        if key not in self._cache:
            p = self._lookup(pid)
            race = RACES[p.race] if p is not None else "human"
            rng = self._rng(_GIVEN, pid)
            self._cache[key] = self._word(rng, race, 2)
        return self._cache[key]

    def surname(self, pid: int) -> str:
        """Inherited from the father, else the mother; a founder or commoner gets one of their own."""
        key = ("s", pid)
        if key not in self._cache:
            p = self._lookup(pid)
            parent = None if p is None else (p.father_id if p.father_id is not None else p.mother_id)
            if parent is not None and self._lookup(parent) is not None:
                self._cache[key] = self.surname(parent)
            else:
                race = RACES[p.race] if p is not None else "human"
                self._cache[key] = self._word(self._rng(_SURNAME, pid), race, 2)
        return self._cache[key]

    def person(self, pid: int) -> str:
        return f"{self.given(pid)} {self.surname(pid)}"

    def house(self, house_id: int) -> str:
        return "House " + self._word(self._rng(_HOUSE, house_id), ("human", "goliath", "elf")[house_id % 3], 2)

    def team(self, team_id: int) -> str:
        rng = self._rng(_TEAM, 0)                              # one shuffle for the whole league: names never repeat
        cities, mascots = rng.permutation(len(_CITIES)), rng.permutation(len(_MASCOTS))
        return f"{_CITIES[cities[team_id % len(cities)]]} {_MASCOTS[mascots[team_id % len(mascots)]]}"
