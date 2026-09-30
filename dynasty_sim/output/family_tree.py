"""Family trees for a house, as indented Markdown and as Graphviz DOT."""
from __future__ import annotations

from dynasty_sim.core.types import RACES

_COLOR = {"human": "#cfe3f7", "goliath": "#f5d5b8", "elf": "#d8f0d0"}


def _members(r, hid: int, depth: int = 4) -> set:
    """Current and former house members plus their ancestors up to `depth` generations."""
    pop = r.world.pop
    ids = {p.id for p in pop.people.values() if p.house_id == hid}
    frontier = set(ids)
    for _ in range(depth):
        nxt = set()
        for pid in frontier:
            p = r.person(pid)
            for par in (p.mother_id, p.father_id) if p is not None else ():
                if par is not None and r.person(par) is not None and par not in ids:
                    nxt.add(par)
        ids |= nxt
        frontier = nxt
    return ids


def family_markdown(r, hid: int, prefix: str = "../") -> str:
    """Descendant tree: each root (no known parent in the set) with children indented beneath."""
    ids = _members(r, hid)
    kids = r.world.pop.children
    roots = sorted((i for i in ids if not any(par in ids for par in _parents(r, i))), key=lambda i: (r.person(i).birth_year, i))
    out = [f"# Family tree of {r.names.house(hid)}", ""]
    seen = set()

    def walk(pid: int, level: int) -> None:
        if pid in seen:
            return
        seen.add(pid)
        p = r.person(pid)
        tag = []
        if p.genome.F > 1e-9:
            tag.append(f"F={p.genome.F:.3f}")
        if not p.alive:
            tag.append(f"died {r.fys(p.death_year)}")
        out.append("  " * level + f"- {r.p(pid, prefix)} (b. {r.fys(p.birth_year)}, {r.race_name(pid)})" + (f" — {', '.join(tag)}" if tag else ""))
        for c in sorted(kids.get(pid, []), key=lambda c: (r.person(c).birth_year, c)):
            if c in ids:
                walk(c, level + 1)

    for root in roots:
        walk(root, 0)
    return "\n".join(out) + "\n"


def _parents(r, pid: int) -> list:
    p = r.person(pid)
    return [x for x in (p.mother_id, p.father_id) if x is not None]


def family_dot(r, hid: int) -> str:
    ids = _members(r, hid)
    lines = [f'digraph "{r.names.house(hid)}" {{', "  rankdir=TB;", '  node [shape=box, style=filled, fontname="Helvetica"];']
    for pid in sorted(ids):
        p = r.person(pid)
        race = RACES[p.race]
        f = f"\\nF={p.genome.F:.3f}" if p.genome.F > 1e-9 else ""
        lines.append(f'  p{pid} [label="{r.names.person(pid)}\\n(b. {r.fys(p.birth_year)}){f}", fillcolor="{_COLOR[race]}"];')
    for pid in sorted(ids):
        for par in _parents(r, pid):
            if par in ids:
                lines.append(f"  p{par} -> p{pid};")
    lines.append("}")
    return "\n".join(lines) + "\n"
