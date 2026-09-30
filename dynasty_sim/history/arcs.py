"""Arcs: a protagonist's chain of notable events. Overlapping arcs are fine (a title belongs to every player and house in it)."""
from __future__ import annotations

from dataclasses import dataclass

from dynasty_sim.history.events import NOTABLE, HistoryEvent


@dataclass
class Arc:
    protagonist: tuple                  # ("person", pid) or ("house", house_id)
    events: list

    @property
    def kind(self) -> str:
        return self.protagonist[0]

    @property
    def span(self) -> tuple:
        return self.events[0].year, self.events[-1].year


def build_arcs(events: list[HistoryEvent], min_events: int = 3, min_importance: int = NOTABLE) -> list[Arc]:
    """Group notable events by protagonist; keep chains of at least `min_events`, ordered by time then id."""
    chains: dict = {}
    for e in events:
        if e.importance < min_importance:
            continue
        for p in e.protagonists():
            chains.setdefault(p, []).append(e)
    arcs = [Arc(p, sorted(es, key=lambda e: (e.year, e.id))) for p, es in chains.items() if len(es) >= min_events]
    return sorted(arcs, key=lambda a: (-len(a.events), a.protagonist))


def linked(arc: Arc) -> bool:
    """True if every event after the first names an earlier event of the same protagonist in its context."""
    ids = [e.id for e in arc.events]
    return all(any(c in ids[:i] for c in e.context) for i, e in enumerate(arc.events) if i > 0)
