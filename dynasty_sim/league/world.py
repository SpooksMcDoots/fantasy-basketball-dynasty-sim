"""World = population + league + per-year dashboard records. The single entry point for full-history runs."""
from __future__ import annotations

from dynasty_sim.config import Params, load_params
from dynasty_sim.core.rng import RngTree
from dynasty_sim.engine.interface import NumbaEngine
from dynasty_sim.history import dashboard
from dynasty_sim.league.season import League
from dynasty_sim.population.demography import Population


class World:
    def __init__(self, seed: int, P: Params | None = None, eng: NumbaEngine | None = None, history: bool = True):
        self.seed = seed
        self.P = P or load_params()
        self.pop = Population(self.P, RngTree(seed))
        self.pop.seed()
        self.league = League(self.pop, eng)
        self.ledger = None
        if history:
            from dynasty_sim.history.ledger import Ledger
            self.ledger = Ledger(self)
            self.league.observers.append(self.ledger)
        self.league.found()
        self.records: list[dashboard.YearRecord] = []

    @property
    def year(self) -> int:
        return self.pop.year

    def step(self) -> dashboard.YearRecord:
        res = self.league.step_year()
        rec = dashboard.snapshot(self, res)
        self.records.append(rec)
        if self.ledger is not None:
            self.ledger.after_season(self, res)
        return rec

    def run(self, years: int) -> "World":
        for _ in range(years):
            self.step()
        return self
