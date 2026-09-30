"""Person and career pages."""
from __future__ import annotations

from dynasty_sim.core.types import RACES, TRAITS
from dynasty_sim.engine import schema as S

SEASON_HEAD = ("| Year | Age | Team | G | MPG | PPG | RPG | APG | SPG | BPG | Value |\n"
               "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")


def _parents(r, p, prefix: str) -> str:
    bits = []
    for label, pid in (("Mother", p.mother_id), ("Father", p.father_id)):
        if pid is not None and r.person(pid) is not None:
            bits.append(f"{label}: {r.p(pid, prefix)}")
    return " · ".join(bits)


def render_person(r, pid: int, prefix: str = "../") -> str:
    p = r.person(pid)
    name = r.names.person(pid)
    led = r.ledger
    out = [f"# {name}", ""]
    race = r.race_name(pid)
    hyb = " (hybrid)" if p.genome.ancestry.max() < 0.999 else ""
    bio = [f"{race}{hyb}", f"born {r.fy(p.birth_year)}"]
    if not p.alive and p.death_year is not None and p.death_year <= r.world.year:
        bio.append(f"died {r.fy(p.death_year)}")
    if p.house_id is not None:
        bio.append(f"of {r.h(p.house_id, prefix)}")
    out += [" · ".join(bio), ""]
    fam = _parents(r, p, prefix)
    if fam:
        out += [fam, ""]
    kids = [c for c in r.world.pop.children.get(pid, []) if r.person(c) is not None]
    if kids:
        out += ["Children: " + ", ".join(r.p(c, prefix) for c in kids), ""]
    if p.genome.F > 1e-9:
        out += [f"Inbreeding coefficient: {p.genome.F:.4f}", ""]
    c = led.careers.get(pid)
    if c is not None and c.seasons:
        out += ["## Career", ""]
        if c.hof_year is not None:
            out += [f"**Hall of Fame, class of year {c.hof_year}.**", ""]
        out += [SEASON_HEAD]
        for s in c.seasons:
            g = max(s.games, 1)
            st = s.stats
            out.append(f"| {s.year} | {s.age} | {r.t(s.team)} | {s.games} | {st[S.ST_SEC] / 60 / g:.1f} | {st[S.ST_PTS] / g:.1f} | "
                       f"{(st[S.ST_ORB] + st[S.ST_DRB]) / g:.1f} | {st[S.ST_AST] / g:.1f} | {st[S.ST_STL] / g:.1f} | "
                       f"{st[S.ST_BLK] / g:.1f} | {s.value:.1f} |")
        out += ["", f"Career value: {c.career_value():.1f} points added over {c.games()} games.", ""]
        if c.awards:
            out += ["## Honours", ""] + [f"- Year {y}: {a}" for y, a in c.awards] + [""]
        if c.exposed:
            out += [f"Congenital traits: {', '.join(c.exposed)}.", ""]
    evs = [e for e in led.log.notable() if pid in e.actors]
    if evs:
        out += ["## Timeline", ""] + [f"- Year {e.year}: {r.narrate(e, prefix)}" for e in sorted(evs, key=lambda e: (e.year, e.id))]
        out.append("")
    return "\n".join(out)
