"""Population state and the yearly demographic step: deaths, marriage, births, immigration."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.ids import IdAllocator
from dynasty_sim.core.rng import RngTree
from dynasty_sim.core.types import RACE_IDX, RACES, Event, House, Person
from dynasty_sim.genetics.inheritance import birth, conception_prob, viability_mortality
from dynasty_sim.genetics.pedigree import Pedigree
from dynasty_sim.population.commoners import draw_death_year, immigrate, spawn_adult
from dynasty_sim.population.succession import resolve_head


def _athletic_score(p: Person) -> float:
    """Best positional draft composite of a person's own-population z-scores (heritable athletic value)."""
    from dynasty_sim.core.reference import ARCHETYPES, draft_z_score
    z = p.genome.A + p.genome.E_perm
    return max(float(draft_z_score(z, a)) for a in range(len(ARCHETYPES)))


def pick_by_score(scores: np.ndarray, lam: float, rng: np.random.Generator) -> int:
    """Index of the chosen candidate: softmax weight exp(lam * score); lam = 0 is a uniform choice."""
    w = np.exp(lam * (scores - scores.max()))
    return int(rng.choice(len(scores), p=w / w.sum()))


class Population:
    def __init__(self, P: Params, tree: RngTree):
        self.P, self.tree = P, tree
        self.people: dict[int, Person] = {}
        self.children: dict[int, list[int]] = defaultdict(list)
        self.houses: dict[int, House] = {}
        self.events: list[Event] = []
        self.ped, self.ids = Pedigree(), IdAllocator()
        self.year = 0

    # ---- queries -------------------------------------------------------------
    def alive_people(self) -> list[Person]:
        return [p for p in self.people.values() if p.alive]

    def race_counts(self) -> np.ndarray:
        c = np.zeros(len(RACES))
        for p in self.people.values():
            if p.alive:
                c[p.race] += 1
        return c

    def living_houses(self) -> list[House]:
        return [h for h in self.houses.values() if h.extinct_year is None]

    def stats(self) -> dict:
        alive = self.alive_people()
        return {"year": self.year, "n": len(alive),
                "by_race": {r: int(c) for r, c in zip(RACES, self.race_counts())},
                "houses_alive": len(self.living_houses()),
                "mean_F": float(np.mean([p.genome.F for p in alive])) if alive else 0.0}

    # ---- setup ---------------------------------------------------------------
    def seed(self) -> None:
        """Year 0: founding houses (couples of one race) plus a commoner fill to seed_fill * K."""
        d, g = self.tree.season("demography", 0), self.tree.season("genetics", 0)
        n_found = self.P.demo["house_founders"]
        for hid, race_name in enumerate(self.P.demo["house_races"], start=1):
            race = RACE_IDX[race_name]
            house = House(hid, f"House {hid}", race)
            genomes = [None] * n_found
            if self.P.league["house_selected"]:
                from dynasty_sim.core.reference import frozen_reference
                from dynasty_sim.league.pool import selected_founders
                genomes = selected_founders(n_found, race_name, g, self.P, frozen_reference())
            members = [spawn_adult(self, race, 0, d, g, sex="F" if k % 2 == 0 else "M", house_id=hid,
                                   genome=genomes[k]) for k in range(n_found)]
            for w, m in zip(members[0::2], members[1::2]):
                w.spouse_id, m.spouse_id = m.id, w.id
            men = [p for p in members if p.sex == "M"]
            house.head_id = min(men, key=lambda p: (p.birth_year, p.id)).id
            self.houses[hid] = house
        counts = self.race_counts()
        for race in range(len(RACES)):
            for _ in range(max(0, int(self.P.seed_fill_count(race) - counts[race]))):
                spawn_adult(self, race, 0, d, g)

    # ---- yearly step ---------------------------------------------------------
    def step(self) -> None:
        self.year += 1
        y = self.year
        d, g = self.tree.season("demography", y), self.tree.season("genetics", y)
        self._deaths(y)
        self._marriages(y, d)
        self._births(y, d, g)
        immigrate(self, y, d, g)

    def _deaths(self, y: int) -> None:
        dead_heads = []
        for p in list(self.people.values()):
            if p.alive and p.death_year is not None and p.death_year <= y:
                p.alive = False
                if p.spouse_id is not None:
                    self.people[p.spouse_id].spouse_id = None
                    p.spouse_id = None
                h = self.houses.get(p.house_id) if p.house_id is not None else None
                if h is not None and h.extinct_year is None and h.head_id == p.id:
                    dead_heads.append((h, p))
                elif h is not None and self._young(p, y):
                    self.events.append(Event(y, "heir_death" if self._is_heir(p, h) else "young_death", h.id, (p.id,),
                                             f"age {y - p.birth_year}"))
        for h, p in dead_heads:
            resolve_head(self, h, p, y)

    def _young(self, p: Person, y: int) -> bool:
        from dynasty_sim.development.aging import age_eff
        return self.P.adult_age[p.race] <= y - p.birth_year and age_eff(y - p.birth_year, p.race, self.P) < 30

    def _is_heir(self, p: Person, house) -> bool:
        """The eldest living child of the current head is the heir presumptive."""
        head = house.head_id
        if head is None or p.id not in self.children[head]:
            return False
        kids = [self.people[c] for c in self.children[head] if self.people[c].house_id == house.id
                and (self.people[c].alive or c == p.id)]
        return bool(kids) and min(kids, key=lambda k: (k.birth_year, k.id)).id == p.id

    def _is_head(self, p: Person) -> bool:
        return p.house_id is not None and self.houses[p.house_id].head_id == p.id

    def _marriages(self, y: int, d: np.random.Generator) -> None:
        P, demo = self.P, self.P.demo
        alive = self.alive_people()
        age = lambda p: y - p.birth_year
        women = [p for p in alive if p.sex == "F" and p.spouse_id is None
                 and P.adult_age[p.race] <= age(p) <= (p.genome.ancestry @ P.fertile_female)[1]]
        men = [p for p in alive if p.sex == "M" and p.spouse_id is None and age(p) >= P.adult_age[p.race]]
        taken: set[int] = set()
        for w in (women[i] for i in d.permutation(len(women))):
            if d.random() > demo["marriage_prob"]:
                continue
            cross = d.random() < demo["p_cross_race"]
            gap = 0.2 * float(w.genome.ancestry @ P.lifespan_mu)
            dynastic = w.house_id is not None and d.random() < demo["p_dynasty_match"]
            pool = [m for m in men if m.id not in taken and (cross or m.race == w.race)
                    and (m.house_id is not None or not dynastic)
                    and abs(age(m) - age(w)) <= gap and not (self._is_head(w) and self._is_head(m))]
            if not pool:
                continue
            ok = []
            for i in d.permutation(len(pool))[: demo["n_candidates"]]:
                m = pool[int(i)]
                F = self.ped.coancestry(w.id, m.id)
                if F >= demo["kin_F_threshold"] and (F >= 0.25 or d.random() < demo["kin_avoid_prob"]):
                    continue
                ok.append(m)
            if ok:
                # assortative pressure: prefer athletic spouses (softmax on the draft composite of their genes)
                m = ok[pick_by_score(np.array([_athletic_score(m) for m in ok]), demo["mate_choice_lambda"], d)]
                self._marry(w, m)
                taken.add(m.id)

    def _marry(self, w: Person, m: Person) -> None:
        hw, hm = w.house_id, m.house_id
        if hw is not None and hm is not None and hw != hm:
            self.events.append(Event(self.year, "marriage", hw, (w.id, m.id), f"{hw}-{hm}"))
        w.spouse_id, m.spouse_id = m.id, w.id
        if self._is_head(w):
            m.house_id = w.house_id
        elif self._is_head(m):
            w.house_id = m.house_id
        else:
            w.house_id = m.house_id = m.house_id if m.house_id is not None else w.house_id

    def _births(self, y: int, d: np.random.Generator, g: np.random.Generator) -> None:
        P, demo = self.P, self.P.demo
        counts = self.race_counts()
        mothers = [p for p in self.alive_people() if p.sex == "F" and p.spouse_id is not None]
        for m in mothers:
            f = self.people[m.spouse_id]
            r = m.race
            if m.last_birth_year is not None and y - m.last_birth_year < P.birth_interval[r]:
                continue
            F = self.ped.coancestry(m.id, f.id)
            p = conception_prob(m.genome, f.genome, y - m.birth_year, P, child_F=F)
            p *= max(0.0, 1.0 - counts[r] / (demo["fertility_zero_at"] * P.K[r]))
            if d.random() >= p:
                continue
            c = birth(m, f, self.ids.next(), y, g, P, self.ped)
            c.house_id = f.house_id if f.house_id is not None else m.house_id
            adult = int(c.genome.ancestry @ P.adult_age)
            if d.random() < demo["child_mortality_base"] + P.child_mortality_per_F * c.genome.F + viability_mortality(c.genome, P):
                c.death_year = y + int(d.integers(0, adult))
            else:
                c.death_year = draw_death_year(c.genome.ancestry, y, 0, d, P)
            self.people[c.id] = c
            self.children[m.id].append(c.id)
            self.children[f.id].append(c.id)
            m.last_birth_year = y
            counts[c.race] += 1
