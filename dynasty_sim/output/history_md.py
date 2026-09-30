"""League-level pages: season-by-season history, records, Hall of Fame, rivalries, arcs and house pages."""
from __future__ import annotations

from dynasty_sim.core.types import RACES
from dynasty_sim.history import events as EV
from dynasty_sim.history.arcs import build_arcs
from dynasty_sim.history.records import PLAYER_CAREER, PLAYER_GAME, PLAYER_SEASON
from dynasty_sim.output import family_tree
from dynasty_sim.output.render import RACE_LABEL, STAT_LABEL


def _game_links(r, year: int, prefix: str = "") -> str:
    led, out = r.ledger, []
    snaps = [s for s in led.snapshots if s.year == year]
    playoff = [s for s in snaps if s.playoff]
    high = [s for s in snaps if not s.playoff]
    if playoff:
        r.need_games[f"games/{year}_final.md"] = (playoff[-1], f"Year {year}: title-deciding game")
        out.append(f"[title-deciding game]({prefix}games/{year}_final.md)")
    if high:
        r.need_games[f"games/{year}_high.md"] = (high[0], f"Year {year}: highest-scoring individual game")
        out.append(f"[top scoring game]({prefix}games/{year}_high.md)")
    return " · ".join(out)


def render_history(r) -> str:
    w, led = r.world, r.ledger
    log = led.log
    out = ["# League history", "", f"{len(w.league.history)} seasons, {len(w.league.teams)} teams.", ""]
    out += ["| Year | Champion | Finalist | MVP | Best record |", "|---:|---|---|---|---|"]
    for res in w.league.history:
        y = res.year
        mvp = led.awards.get(y, {}).get("mvp")
        best = int(res.wins.argmax())
        out.append(f"| [{y}](#year-{y}) | {r.team_house(res.champion)} | {r.t(res.finalist)} | "
                   f"{r.p(mvp) if mvp is not None else '—'} | {r.t(best)} ({res.wins[best]}-{res.losses[best]}) |")
    out.append("")
    by_year: dict = {}
    for e in log.notable():
        by_year.setdefault(e.year, []).append(e)
    for res in w.league.history:
        y = res.year
        out += [f"## Year {y}", "", f"**{r.team_house(res.champion)}** won the title over {r.t(res.finalist)}. "
                f"Regular-season best: {r.t(int(res.wins.argmax()))} ({res.wins.max()}-{res.losses[res.wins.argmax()]}).", ""]
        links = _game_links(r, y)
        if links:
            out += [links, ""]
        aw = led.awards.get(y, {})
        if aw.get("all_league"):
            out += ["All-League first team: " + ", ".join(r.p(p) for p in aw["all_league"]) + ".", ""]
        evs = sorted((e for e in by_year.get(y, []) if e.importance >= EV.MAJOR or (e.importance == EV.NOTABLE and e.kind not in ("award", "marriage"))),
                     key=lambda e: (-e.importance, e.id))
        evs = [e for e in evs if not (e.kind == "award" and e.detail.get("award") == "All-League")]
        out += [f"- {r.narrate(e)}" for e in evs[:10]]
        if len(evs) > 10:
            out.append(f"- …and {len(evs) - 10} smaller items.")
        out.append("")
    return "\n".join(out)


def render_records(r) -> str:
    book = r.ledger.records
    out = ["# Record book", ""]
    groups = [("Single game", "game", PLAYER_GAME), ("Single season (40+ games)", "season", PLAYER_SEASON),
              ("Career", "career", PLAYER_CAREER)]
    for title, kind, stats in groups:
        out += [f"## {title}", "", "| Statistic | League | Human | Goliath | Elf |", "|---|---|---|---|---|"]
        for stat in stats:
            cells = []
            for scope in ("all", "human", "goliath", "elf"):
                rec = book.holder(kind, stat, scope)
                cells.append(f"{rec.value:g} — {r.p(rec.holder)} ({rec.year})" if rec else "—")
            out.append(f"| {STAT_LABEL.get(stat, stat)} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Teams", "", "| Record | Team | Value | Year |", "|---|---|---:|---:|"]
    for kind, stat, direction, label in (("team_game", "team_pts", "max", "Most points in a game"), ("team_game", "team_pts", "min", "Fewest points in a game"),
                                          ("team_game", "team_margin", "max", "Largest margin of victory"), ("team_season", "wins", "max", "Most wins"),
                                          ("team_season", "wins", "min", "Fewest wins"), ("team_season", "mov", "max", "Best season margin")):
        rec = book.holder(kind, stat, "all", direction)
        if rec:
            out.append(f"| {label} | {r.team_house(rec.holder)} | {rec.value:g} | {rec.year} |")
    return "\n".join(out) + "\n"


def render_hof(r) -> str:
    led = r.ledger
    out = ["# Hall of Fame", "", "## Players", "", "| Class | Player | Race | Seasons | Career value |", "|---:|---|---|---:|---:|"]
    for pid, year in sorted(led.hof.players.items(), key=lambda kv: (kv[1], kv[0])):
        c = led.careers[pid]
        out.append(f"| {year} | {r.p(pid)} | {r.race_name(pid)} | {len(c.seasons)} | {c.career_value():.0f} |")
    out += ["", "## Dynasty builders", ""]
    if led.hof.builders:
        out += ["| Class | Head | House | Titles |", "|---:|---|---|---:|"]
        for pid, (year, hid, titles) in sorted(led.hof.builders.items(), key=lambda kv: (kv[1][0], kv[0])):
            out.append(f"| {year} | {r.p(pid)} | {r.h(hid)} | {titles} |")
    else:
        out.append("No head of house has yet won three titles in a tenure.")
    return "\n".join(out) + "\n"


def render_rivalries(r) -> str:
    rv = r.ledger.rivalry
    out = ["# Rivalries", "",
           f"A pairing is a rivalry once its index reaches {rv.relative:.2f}× the median pairing (now {rv.level:.0f}). "
           "Games alone put every pairing near the same level, so what separates rivals is playoffs, close games, "
           "moves and family ties. The index decays 15% each season.", "",
           "| Rivalry | Peak index | Peak year | Now | Main sources |", "|---|---:|---:|---:|---|"]
    rows = sorted(rv.peak.items(), key=lambda kv: (-kv[1][0], kv[0]))[:12]
    for (a, b), (peak, year) in rows:
        out.append(f"| {r.h(a)} vs {r.h(b)} | {peak:.0f} | {year} | {rv.index[(a, b)]:.0f} | " + ", ".join(rv.why((a, b))) + " |")
    out += ["", "## Hooks by season (top three, once flagged)", ""]
    for year in sorted(r.ledger.rivalry_top):
        live = [(k, v) for k, v in r.ledger.rivalry_top[year] if k in rv.flagged and rv.flagged[k] <= year]
        if live:
            out.append(f"- Year {year}: " + "; ".join(f"{r.h(a)}–{r.h(b)} ({v:.0f})" for (a, b), v in live))
    return "\n".join(out) + "\n"


def render_arcs(r, limit: int = 40) -> str:
    arcs = build_arcs(r.ledger.log.events)
    out = ["# Arcs", "", f"{len(arcs)} storylines of three or more linked events; the longest {min(limit, len(arcs))} follow.", ""]
    for arc in arcs[:limit]:
        who = r.p(arc.protagonist[1]) if arc.kind == "person" else r.h(arc.protagonist[1])
        y0, y1 = arc.span
        out += [f"## {who} — years {y0}–{y1}", ""] + [f"- Year {e.year}: {r.narrate(e)}" for e in arc.events] + [""]
    return "\n".join(out)


def render_house(r, hid: int) -> str:
    w, led = r.world, r.ledger
    house = w.pop.houses[hid]
    team = next(t for t in w.league.teams if t.house_id == hid)
    titles = led.titles.get(hid, [])
    out = [f"# {r.names.house(hid)}", "", f"{RACE_LABEL[RACES[house.race]]} house, founded year {house.founded_year}. "
           f"Team: {r.t(team.id)}.", ""]
    if house.extinct_year is not None:
        out += [f"The house died out in year {house.extinct_year}.", ""]
    elif house.head_id is not None:
        out += [f"Current head: {r.p(house.head_id, '../')}.", ""]
    out += [f"Titles: {len(titles)}" + (f" ({', '.join(str(y) for y in titles)})" if titles else ""), ""]
    evs = [e for e in led.log.notable() if hid in e.houses and e.kind in ("succession", "adoption", "extinction", "heir_death", "succession_effect", "hof_builder", "rivalry", "drought_ended")]
    if evs:
        out += ["## Dynasty events", ""] + [f"- Year {e.year}: {r.narrate(e, '../')}" for e in sorted(evs, key=lambda e: (e.year, e.id))] + [""]
    out += ["## Family tree", "", f"[Graphviz version](./{hid}.dot)", ""]
    out += family_tree.family_markdown(r, hid).splitlines()[2:]
    return "\n".join(out) + "\n"


def render_index(r) -> str:
    w = r.world
    return "\n".join([
        "# Dynasty basketball history", "",
        f"Seed {w.seed}, {len(w.league.history)} seasons.", "",
        "- [League history](history.md)", "- [Record book](records.md)", "- [Hall of Fame](hall_of_fame.md)",
        "- [Rivalries](rivalries.md)", "- [Arcs](arcs.md)", "", "## Houses", "",
        *[f"- {r.h(hid)}" for hid in sorted(w.pop.houses)], ""])
