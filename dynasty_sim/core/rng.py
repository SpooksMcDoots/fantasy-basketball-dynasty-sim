"""Seed tree: root SeedSequence -> named subsystem streams -> per-season / per-game seeds.

Everything is derived from explicit spawn_keys (never the stateful spawn()), so the
order in which subsystems run can never change what another subsystem draws.
"""
from __future__ import annotations

import numpy as np

STREAMS = ("genetics", "demography", "league", "engine", "narrative")


class RngTree:
    def __init__(self, run_seed: int):
        self.run_seed = int(run_seed)

    def _seq(self, stream: str, *path: int) -> np.random.SeedSequence:
        return np.random.SeedSequence(self.run_seed, spawn_key=(STREAMS.index(stream), *path))

    def stream(self, stream: str) -> np.random.Generator:
        """Whole-run generator for a subsystem."""
        return np.random.default_rng(self._seq(stream))

    def season(self, stream: str, year: int) -> np.random.Generator:
        """Independent generator for (subsystem, year)."""
        return np.random.default_rng(self._seq(stream, int(year)))

    def game_seed(self, season: int, game_id: int) -> int:
        """uint32 seed for the numba kernel, derived from the engine stream."""
        return int(self._seq("engine", int(season), int(game_id)).generate_state(1, np.uint32)[0])

    def manifest(self) -> dict:
        return {"run_seed": self.run_seed, "streams": list(STREAMS)}
