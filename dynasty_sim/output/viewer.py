"""History viewer: flatten a finished world into one compact JSON payload and wrap it in a static HTML page.

`viewer.json` is the saved run the page can load; `viewer.html` is the same page with that run embedded, so it opens
by double-click. The page itself (`viewer.html` template next to this file) needs no server and no network.
"""
from __future__ import annotations

import json
from pathlib import Path

from dynasty_sim.core.types import RACES, TRAITS
from dynasty_sim.engine import schema as S
from dynasty_sim.history import alarms as AL
from dynasty_sim.history import events as EV
from dynasty_sim.history import rivalry as RV
from dynasty_sim.output.render import RACE_LABEL, Renderer

FORMAT = "dynasty-viewer"
VERSION = 2
TEMPLATE = Path(__file__).with_name("viewer_template.html")
AWARD_KEYS = ("mvp", "finals_mvp", "dpoy", "roy", "scoring", "most_improved")


def _r1(x: float) -> float:
    return round(float(x), 1)


def _season_row(s) -> list:
    g, st = max(s.games, 1), s.stats
    return [int(s.year), int(s.age), int(s.team), int(s.games), _r1(st[S.ST_SEC] / 60 / g), _r1(st[S.ST_PTS] / g),
            _r1((st[S.ST_ORB] + st[S.ST_DRB]) / g), _r1(st[S.ST_AST] / g), _r1(st[S.ST_STL] / g), _r1(st[S.ST_BLK] / g),
            _r1(s.value)]


def _players(world, r: Renderer) -> dict:
    led, pop = world.ledger, world.pop
    ids = set(pop.people) | set(led.persons)
    out = {}
    for pid in sorted(ids):
        p = r.person(pid)
        if p is None:
            continue
        c = led.careers.get(pid)
        dead = not p.alive and p.death_year is not None and p.death_year <= world.year
        e = {"n": r.names.person(pid), "r": int(p.race), "hy": bool(p.genome.ancestry.max() < 0.999), "b": int(p.birth_year),
             "d": int(p.death_year) if dead else None, "h": p.house_id,
             "m": p.mother_id if p.mother_id in ids else None, "f": p.father_id if p.father_id in ids else None,
             "k": [k for k in pop.children.get(pid, []) if k in ids]}
        if e["hy"]:
            e["an"] = [round(float(a), 2) for a in p.genome.ancestry]
        if c is not None and c.seasons:
            e["s"] = [_season_row(s) for s in c.seasons]
            e["cv"] = _r1(c.career_value())
            e["ret"] = c.retired_year
            e["hof"] = c.hof_year
            e["aw"] = [[int(y), a] for y, a in c.awards]
            e["tot"] = [int(c.total(S.ST_PTS)), int(c.total(S.ST_ORB) + c.total(S.ST_DRB)), int(c.total(S.ST_AST)),
                        int(c.total(S.ST_STL)), int(c.total(S.ST_BLK)), c.games(), _r1(c.career_value())]   # pts reb ast stl blk g value
            e["tt"] = int(led.player_titles.get(pid, 0))
            e["hg"] = int(c.hyb_gen)
            if c.exposed:
                e["ex"] = list(c.exposed)
        out[str(pid)] = e
    return out


def _events(r: Renderer) -> list:
    out = []
    for e in r.ledger.log.notable():
        if e.kind == "award" and e.detail.get("award") == "All-League":
            continue                                           # shown on the season page instead of the feed
        try:
            text = r.narrate(e)
        except Exception:                                      # an odd event must not stop the export
            text = e.kind.replace("_", " ")
        out.append({"i": e.id, "y": int(e.year), "k": e.kind, "m": int(e.importance), "t": text,
                    "a": list(e.actors), "h": list(e.houses)})
    return out


def _years(world) -> list:
    led = world.ledger
    finals = {}
    for s in led.snapshots:
        if s.playoff:
            finals[s.year] = s                                  # the last playoff snapshot of a year is its deciding game
    out = []
    for res in world.league.history:
        aw = led.awards.get(res.year, {})
        row = {"y": int(res.year), "c": int(res.champion), "f": int(res.finalist),
               "w": [int(x) for x in res.wins], "l": [int(x) for x in res.losses], "mov": [_r1(x) for x in res.mov],
               "seeds": [int(x) for x in res.seeds], "series": [[int(a), int(b), int(w), int(n)] for a, b, w, n in res.series],
               "aw": {k: aw[k] for k in AWARD_KEYS if aw.get(k) is not None}}
        if aw.get("all_league"):
            row["aw"]["all_league"] = [int(p) for p in aw["all_league"]]
        snap = finals.get(res.year)
        if snap is not None:
            row["fin"] = {"teams": [int(t) for t in snap.teams], "score": list(snap.score)}
        out.append(row)
    return out


def _health(world) -> dict:
    """Per-year dashboard series, plus the stability alarms and their thresholds."""
    recs = world.records
    rnd = lambda a, d=3: [round(float(x), d) for x in a]
    metrics = {k: [round(float(r.metrics[k]), 4) for r in recs] for k in recs[0].metrics} if recs else {}
    alarms = AL.check_alarms(recs) if len(recs) > AL.BURN_IN else []
    return {"years": [int(r.year) for r in recs], "traits": list(TRAITS), "f": [round(float(r.mean_F), 5) for r in recs],
            "top_mov": [round(float(r.top_mov), 2) for r in recs], "win_sd": [round(float(r.win_sd), 4) for r in recs],
            "champ_house": [int(r.champion_house) for r in recs], "dyn": [round(float(r.dynasty_share), 3) for r in recs],
            "houses": [int(r.houses_alive) for r in recs], "race_share": [rnd(r.race_share) for r in recs],
            "pop_race": [[int(x) for x in r.pop_by_race] for r in recs], "mz": [rnd(r.league_mean_z) for r in recs],
            "pz": [rnd(r.pop_mean_z) for r in recs], "p99": [rnd(r.pop_p99_z) for r in recs], "metrics": metrics,
            "alarms": [{"kind": a.kind, "detail": a.detail} for a in alarms],
            "limits": {"burn_in": AL.BURN_IN, "mean_drift": AL.MEAN_DRIFT, "p99_drift": AL.P99_DRIFT, "top_mov": AL.TOP_MOV,
                       "title_share": AL.TITLE_SHARE, "title_window": AL.TITLE_WINDOW}}


def _records(world, team_of_house: dict) -> dict:
    """Current holders and every recorded break (including the small per-race ones the feed leaves out)."""
    led = world.ledger
    cur = [{"k": k, "s": s, "sc": sc, "d": d, "v": round(e.value, 2), "p": int(e.holder), "y": int(e.year)}
           for (k, s, sc, d), e in sorted(led.records.best.items())]
    log = []
    for e in led.log.by_kind("record"):
        d = e.detail
        team = d["record_kind"].startswith("team")
        log.append({"y": int(e.year), "k": d["record_kind"], "s": d["stat"], "sc": d["scope"], "d": d["direction"],
                    "v": round(float(e.value), 2), "o": round(float(d["old_value"]), 2), "op": int(d["old_holder"]),
                    "oy": int(d["old_year"]), "p": team_of_house[e.houses[0]] if team else int(e.actors[0])})
    return {"best": cur, "log": log}


def _rivalry(world) -> dict:
    led = world.ledger
    rv, years = led.rivalry, [int(r.year) for r in world.league.history]
    pairs = []
    for (a, b), v in sorted(rv.index.items()):
        peak = rv.peak.get((a, b), (0.0, 0))
        pairs.append({"a": int(a), "b": int(b), "i": round(float(v), 1), "pk": round(float(peak[0]), 1), "py": int(peak[1]),
                      "fl": rv.flagged.get((a, b)), "rs": {k: round(float(x)) for k, x in rv.reasons[(a, b)].items()}})
    hist = {f"{a}-{b}": [round(float(led.rivalry_hist[y][0].get((a, b), 0.0)), 1) if y in led.rivalry_hist else None for y in years]
            for (a, b) in rv.index}
    return {"pairs": pairs, "hist": hist, "years": years,
            "level": [round(float(led.rivalry_hist[y][1]), 1) if y in led.rivalry_hist else None for y in years],
            "top": {str(y): [[int(a), int(b), round(float(v), 1)] for (a, b), v in t] for y, t in led.rivalry_top.items()},
            "relative": RV.RELATIVE_THRESHOLD, "min_level": RV.MIN_LEVEL, "sustained": RV.SUSTAINED}


def build_payload(world) -> dict:
    """Everything the page shows, as JSON-ready Python. Needs a world run with history on."""
    if world.ledger is None:
        raise ValueError("the viewer needs a world run with history=True")
    led, lg, r = world.ledger, world.league, Renderer(world)
    teams = [{"id": t.id, "name": r.t(t.id), "house": t.house_id} for t in lg.teams]
    team_of_house = {t["house"]: t["id"] for t in teams}
    houses = {str(hid): {"n": led.names.house(hid), "r": int(h.race), "fo": int(h.founded_year),
                         "ex": None if h.extinct_year is None else int(h.extinct_year), "t": [int(y) for y in led.titles.get(hid, [])],
                         "team": team_of_house.get(hid)} for hid, h in world.pop.houses.items()}
    hof = [{"p": int(pid), "y": int(y), "career": _r1(led.hof.record[pid]["career"]), "peak": _r1(led.hof.record[pid]["peak"]),
            "honours": _r1(led.hof.record[pid]["honours"]), "score": round(led.hof.record[pid]["score"], 2),
            "role": led.hof.record[pid].get("role")}
           for pid, y in led.hof.players.items()]
    builders = [{"p": int(pid), "y": int(y), "h": int(hid), "t": int(t)} for pid, (y, hid, t) in led.hof.builders.items()]
    return {"format": FORMAT, "version": VERSION,
            "meta": {"seed": int(world.seed), "years": len(lg.history), "teams": len(teams), "people": len(world.pop.people),
                     "first_year": int(lg.history[0].year) if lg.history else 0, "last_year": int(lg.history[-1].year) if lg.history else 0},
            "races": [RACE_LABEL[n] for n in RACES], "teams": teams, "houses": houses, "years": _years(world),
            "events": _events(r), "players": _players(world, r), "hof": hof, "builders": builders,
            "health": _health(world), "records": _records(world, team_of_house), "rivalry": _rivalry(world)}


def dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def render_html(payload: dict | None = None) -> str:
    """The page, with `payload` embedded when given (otherwise it waits for a file or a server)."""
    data = "null" if payload is None else dumps(payload).replace("</", "<\\/").replace("<!--", "<\\u0021--")
    return TEMPLATE.read_text(encoding="utf-8").replace("__EMBEDDED_RUN__", data)


def write_viewer(world, out: Path) -> dict:
    """Write viewer.json (loadable saved run) and viewer.html (the page with the run embedded) into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    payload = build_payload(world)
    (out / "viewer.json").write_text(dumps(payload), encoding="utf-8")
    (out / "viewer.html").write_text(render_html(payload), encoding="utf-8")
    return payload
