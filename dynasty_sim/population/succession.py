"""Heir selection: child -> sibling -> nephew/niece -> closest kin in house -> cadet adoption -> extinction."""
from __future__ import annotations

from typing import Optional

from dynasty_sim.core.types import Event, House, Person


def _age(p: Person, year: int) -> int:
    return year - p.birth_year


def _eldest(cands: list[Person]) -> Optional[Person]:
    return min(cands, key=lambda p: (p.birth_year, p.id)) if cands else None


def resolve_head(pop, house: House, dead_head: Person, year: int) -> None:
    members = [p for p in pop.alive_people() if p.house_id == house.id and p.id != dead_head.id]

    def kids_in_house(pid: int) -> list[Person]:
        return [pop.people[c] for c in pop.children[pid]
                if pop.people[c].alive and pop.people[c].house_id == house.id]

    heir, how = _eldest(kids_in_house(dead_head.id)), "child"
    if heir is None:
        sibs = [p for p in members if _shares_parent(p, dead_head)]
        heir, how = _eldest(sibs), "sibling"
        if heir is None:
            nephews = [k for s in _siblings_any(pop, dead_head) for k in kids_in_house(s.id)]
            heir, how = _eldest(nephews), "nephew"
    if heir is None and members:
        best = max(pop.ped.coancestry(dead_head.id, p.id) for p in members)
        heir, how = _eldest([p for p in members
                             if abs(pop.ped.coancestry(dead_head.id, p.id) - best) < 1e-12]), "kin"
    if heir is None:
        heir = _adopt_cadet(pop, house, dead_head, year)
        how = "cadet"
    if heir is None:
        house.head_id, house.extinct_year = None, year
        pop.events.append(Event(year, "extinction", house.id, (dead_head.id,), "no living kin"))
        return
    house.head_id = heir.id
    detail = how if how != "cadet" else f"cadet from house {getattr(pop, '_last_cadet_origin', None)}"
    pop.events.append(Event(year, "succession" if how != "cadet" else "adoption", house.id,
                            (dead_head.id, heir.id), detail))


def _shares_parent(a: Person, b: Person) -> bool:
    return a.id != b.id and ((a.mother_id is not None and a.mother_id == b.mother_id)
                             or (a.father_id is not None and a.father_id == b.father_id))


def _siblings_any(pop, p: Person) -> list[Person]:
    """Living or dead siblings, so nephews via a dead sibling still count."""
    out = {}
    for parent in (p.mother_id, p.father_id):
        if parent is not None:
            for c in pop.children[parent]:
                if c != p.id:
                    out[c] = pop.people[c]
    return list(out.values())


def _adopt_cadet(pop, house: House, dead_head: Person, year: int) -> Optional[Person]:
    """Closest living adult kin outside the house takes over, bringing spouse and living children."""
    P = pop.P
    pool = [p for p in pop.alive_people()
            if p.house_id != house.id and _age(p, year) >= P.adult_age[p.race]
            and (p.house_id is None or pop.houses[p.house_id].head_id != p.id)]
    scored = [(pop.ped.coancestry(dead_head.id, p.id), p) for p in pool]
    scored = [(c, p) for c, p in scored if c > 0]
    if not scored:
        return None
    best = max(c for c, _ in scored)
    heir = _eldest([p for c, p in scored if abs(c - best) < 1e-12])
    pop._last_cadet_origin = heir.house_id
    for member in [heir] + [pop.people[c] for c in pop.children[heir.id] if pop.people[c].alive] + \
            ([pop.people[heir.spouse_id]] if heir.spouse_id is not None else []):
        member.house_id = house.id
    return heir
