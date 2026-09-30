"""Write a run's readable history (Markdown + DOT) and optional tables (Parquet); verify every link resolves."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from dynasty_sim.engine import schema as S
from dynasty_sim.history.arcs import build_arcs
from dynasty_sim.output import boxscore_md, career_md, family_tree, history_md
from dynasty_sim.output.render import Renderer

_LINK = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")
_BAD = re.compile(r"\bNone\b|\bnan\b|\{|\}|<id|\?\?|\bpid\b")


def build_pages(world) -> dict[str, str]:
    """All pages keyed by relative path. Player/house/game pages are generated until no new link target appears."""
    r = Renderer(world)
    pages = {"index.md": history_md.render_index(r), "history.md": history_md.render_history(r),
             "records.md": history_md.render_records(r), "hall_of_fame.md": history_md.render_hof(r),
             "rivalries.md": history_md.render_rivalries(r), "arcs.md": history_md.render_arcs(r)}
    for hid in world.pop.houses:
        r.need_houses.add(hid)
    done_p, done_h, done_g = set(), set(), set()
    while True:
        todo_p, todo_h = r.need_players - done_p, r.need_houses - done_h
        todo_g = set(r.need_games) - done_g
        if not (todo_p or todo_h or todo_g):
            break
        for pid in sorted(todo_p):
            pages[f"players/{pid}.md"] = career_md.render_person(r, pid)
        for hid in sorted(todo_h):
            pages[f"houses/{hid}.md"] = history_md.render_house(r, hid)
            pages[f"houses/{hid}.dot"] = family_tree.family_dot(r, hid)
        for path in sorted(todo_g):
            snap, title = r.need_games[path]
            pages[path] = boxscore_md.render_box(r, snap, title)
        done_p |= todo_p
        done_h |= todo_h
        done_g |= todo_g
    return pages


def check_links(out: Path) -> list[str]:
    """Problems found: links to missing files, and placeholder-looking text in any Markdown page."""
    problems = []
    for md in sorted(out.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for target in _LINK.findall(text):
            if "://" not in target and not (md.parent / target).resolve().exists():
                problems.append(f"{md.relative_to(out)}: unresolved link {target}")
        for m in _BAD.finditer(text):
            problems.append(f"{md.relative_to(out)}: placeholder text {m.group(0)!r}")
    return problems


def export_tables(world, out: Path) -> None:
    import pandas as pd
    led = world.ledger
    rows = [{"pid": c.pid, "year": s.year, "age": s.age, "team": s.team, "games": s.games, "value": s.value,
             **{n: float(s.stats[i]) for i, n in enumerate(S.STAT_NAMES)}} for c in led.careers.values() for s in c.seasons]
    pd.DataFrame(rows).to_parquet(out / "player_seasons.parquet", index=False)
    pd.DataFrame([{"id": e.id, "year": e.year, "kind": e.kind, "importance": e.importance, "actors": json.dumps(list(e.actors)),
                   "houses": json.dumps(list(e.houses)), "metric": e.metric, "value": e.value, "percentile": e.percentile,
                   "context": json.dumps(list(e.context))} for e in led.log.events]).to_parquet(out / "events.parquet", index=False)
    res = world.league.history
    pd.DataFrame([{"year": r.year, "team": t, "wins": int(r.wins[t]), "losses": int(r.losses[t]), "mov": float(r.mov[t]),
                   "champion": r.champion == t} for r in res for t in range(len(r.wins))]).to_parquet(out / "team_seasons.parquet", index=False)


def export_history(world, out: Path, parquet: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pages = build_pages(world)
    for rel, text in pages.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    if parquet:
        export_tables(world, out)
    return {"pages": len(pages), "problems": check_links(out), "arcs": len(build_arcs(world.ledger.log.events))}
