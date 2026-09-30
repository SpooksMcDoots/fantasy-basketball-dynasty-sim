"""M7 output layer and acceptance test (d): a retellable 50-year history."""
import json
import re

import numpy as np
import pandas as pd
import pytest

from dynasty_sim.cli import main
from dynasty_sim.history.arcs import build_arcs, linked
from dynasty_sim.league.world import World
from dynasty_sim.output import boxscore_md, export, family_tree
from dynasty_sim.output.render import Renderer


@pytest.fixture(scope="module")
def world50():
    return World(3).run(50)


@pytest.fixture(scope="module")
def exported(world50, tmp_path_factory):
    out = tmp_path_factory.mktemp("history")
    summary = export.export_history(world50, out, parquet=True)
    return out, summary


# ---- acceptance (d) ----------------------------------------------------------------------------------------------
@pytest.mark.acceptance
def test_d_at_least_ten_arcs_of_three_or_more_linked_events(world50):
    arcs = build_arcs(world50.ledger.log.events)
    assert len(arcs) >= 10 and all(len(a.events) >= 3 for a in arcs) and all(linked(a) for a in arcs)
    assert sum(a.kind == "person" for a in arcs) >= 10          # real people, not just houses


@pytest.mark.acceptance
def test_d_at_least_three_rivalries_above_threshold(world50):
    rv = world50.ledger.rivalry
    assert len(rv.flagged) >= 3
    assert all(rv.peak[k][0] >= rv.level * 0.9 for k in rv.flagged)


@pytest.mark.acceptance
def test_d_a_record_is_broken_in_every_decade(world50):
    decades = {(e.year - 1) // 10 for e in world50.ledger.log.by_kind("record")}
    assert decades >= {1, 2, 3, 4}                               # decade 0 sets the baseline books


@pytest.mark.acceptance
def test_d_a_succession_has_an_on_court_consequence(world50):
    eff = world50.ledger.succession_effects
    assert eff and any(e["flagged"] and abs(e["delta"]) > 0.15 for e in eff)


@pytest.mark.acceptance
def test_d_markdown_renders_with_no_unresolved_ids(exported):
    out, summary = exported
    assert summary["problems"] == [], summary["problems"][:5]
    assert summary["pages"] > 100 and (out / "history.md").exists()


# ---- structure of the output -----------------------------------------------------------------------------------------------
def test_every_award_winner_and_house_has_a_page(world50, exported):
    out, _ = exported
    for aw in world50.ledger.awards.values():
        mvp = aw.get("mvp")
        if mvp is not None:
            assert (out / "players" / f"{mvp}.md").exists()
    for hid in world50.pop.houses:
        assert (out / "houses" / f"{hid}.md").exists() and (out / "houses" / f"{hid}.dot").exists()


def test_player_page_lists_seasons_awards_and_family(world50, exported):
    out, _ = exported
    led = world50.ledger
    star = max(led.careers.values(), key=lambda c: c.career_value())
    text = (out / "players" / f"{star.pid}.md").read_text(encoding="utf-8")
    assert text.startswith(f"# {led.names.person(star.pid)}")
    assert text.count("\n| ") >= len(star.seasons)                                     # one row per season
    assert "Career value" in text and (not star.awards or "## Honours" in text)


def test_box_score_totals_match_the_final_score(world50):
    r = Renderer(world50)
    snap = next(s for s in world50.ledger.snapshots if s.playoff)
    md = boxscore_md.render_box(r, snap, "Test game")
    totals = re.findall(r"\*\*Team\*\* \| \d+ \| (\d+) \|", md)
    assert [int(x) for x in totals] == list(snap.score)
    assert md.count("| [") >= 10                                                       # players are linked


def test_family_tree_markdown_and_dot_agree(world50):
    r = Renderer(world50)
    hid = next(iter(world50.pop.houses))
    md, dot = family_tree.family_markdown(r, hid), family_tree.family_dot(r, hid)
    nodes = set(re.findall(r"^  p(\d+) \[", dot, re.M))
    edges = re.findall(r"^  p(\d+) -> p(\d+);", dot, re.M)
    assert nodes and all(a in nodes and b in nodes for a, b in edges)
    linked_ids = set(re.findall(r"players/(\d+)\.md", md))
    assert linked_ids and linked_ids <= nodes
    assert dot.count("{") == 1 and dot.strip().endswith("}")


def test_parquet_tables_match_the_ledger(world50, exported):
    out, _ = exported
    led = world50.ledger
    ps = pd.read_parquet(out / "player_seasons.parquet")
    assert len(ps) == sum(len(c.seasons) for c in led.careers.values())
    assert len(pd.read_parquet(out / "events.parquet")) == len(led.log.events)
    ts = pd.read_parquet(out / "team_seasons.parquet")
    assert len(ts) == 50 * 8 and ts["champion"].sum() == 50


def test_hall_of_fame_rate_is_about_one_and_a_quarter_per_season(world50):
    n = len(world50.ledger.hof.players)
    assert 0.8 <= n / 50 <= 1.4


# ---- reproducibility and isolation ---------------------------------------------------------------------------------------
def test_recording_history_does_not_change_the_simulation():
    a, b = World(5, history=True).run(6), World(5, history=False).run(6)
    for ra, rb in zip(a.league.history, b.league.history):
        assert np.array_equal(ra.wins, rb.wins) and ra.champion == rb.champion
    assert [t.roster for t in a.league.teams] == [t.roster for t in b.league.teams]


def test_cli_history_pages_are_byte_identical_across_runs(tmp_path):
    for name in ("a", "b"):
        main(["run", "--years", "5", "--seed", "11", "--out", str(tmp_path / name)])
    files = sorted(p.relative_to(tmp_path / "a") for p in (tmp_path / "a").rglob("*") if p.is_file())
    assert len(files) > 30
    for f in files:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f
    man = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert man["history"]["unresolved"] == [] and man["history"]["pages"] > 30
