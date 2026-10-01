"""History viewer: the payload is self-consistent, round-trips through JSON, and the page embeds it safely."""
import json
import re

import pytest

from dynasty_sim.league.world import World
from dynasty_sim.output import viewer
from dynasty_sim.output.viewer_server import saved_runs


@pytest.fixture(scope="module")
def world():
    return World(3).run(12)


@pytest.fixture(scope="module")
def payload(world):
    return viewer.build_payload(world)


def test_payload_covers_every_season_and_team(world, payload):
    assert payload["format"] == viewer.FORMAT and payload["meta"]["years"] == 12
    assert [y["y"] for y in payload["years"]] == [r.year for r in world.league.history]
    n = len(payload["teams"])
    for y in payload["years"]:
        assert len(y["w"]) == len(y["l"]) == len(y["mov"]) == n
        assert 0 <= y["c"] < n and y["c"] != y["f"]


def test_every_referenced_person_and_house_exists(payload):
    players, houses = payload["players"], payload["houses"]
    for y in payload["years"]:
        for pid in [p for v in y["aw"].values() for p in (v if isinstance(v, list) else [v])]:
            assert str(pid) in players
    for e in payload["events"]:
        assert all(str(a) in players for a in e["a"]) and all(str(h) in houses for h in e["h"])
        assert "None" not in e["t"] and "nan" not in e["t"].split()
    for p in players.values():
        assert all(str(k) in players for k in p["k"])
        assert p["m"] is None or str(p["m"]) in players


def test_player_seasons_match_the_ledger(world, payload):
    for pid, c in list(world.ledger.careers.items())[:50]:
        rows = payload["players"][str(pid)]["s"]
        assert [r[0] for r in rows] == [s.year for s in c.seasons]
        assert sum(r[10] for r in rows) == pytest.approx(c.career_value(), abs=0.1 * len(rows) + 0.1)


def test_embedded_page_round_trips_and_cannot_close_its_script(payload):
    payload = {**payload, "events": payload["events"] + [{"i": -1, "y": 0, "k": "x", "m": 3, "t": "</script><!-- x", "a": [], "h": []}]}
    html = viewer.render_html(payload)
    assert "__EMBEDDED_RUN__" not in html
    script = re.search(r"const EMBEDDED = (.*?);\n\n/\* -+ helpers", html, re.S).group(1)
    assert "</script" not in script and "<!--" not in script
    assert json.loads(script)["events"][-1]["t"] == "</script><!-- x"
    assert "const EMBEDDED = null;" in viewer.render_html(None)


def test_write_viewer_and_saved_run_listing(world, tmp_path):
    viewer.write_viewer(world, tmp_path / "run1")
    assert (tmp_path / "run1" / "viewer.html").stat().st_size > (tmp_path / "run1" / "viewer.json").stat().st_size
    runs = saved_runs(tmp_path)
    assert [(r["path"], r["seed"], r["years"]) for r in runs] == [("run1/viewer.json", 3, 12)]


def test_health_series_line_up_with_the_seasons(payload):
    h, n = payload["health"], payload["meta"]["years"]
    assert h["years"] == [y["y"] for y in payload["years"]]
    for k in ("f", "top_mov", "win_sd", "champ_house", "dyn", "houses", "race_share", "pop_race", "mz", "pz", "p99"):
        assert len(h[k]) == n, k
    assert all(len(v) == n for v in h["metrics"].values())
    assert len(h["mz"][0]) == len(h["traits"]) and {"top_mov", "title_share", "burn_in"} <= set(h["limits"])
    assert all(str(c) in payload["houses"] for c in h["champ_house"])


def test_record_book_holders_exist(payload):
    players, teams = payload["players"], {t["id"] for t in payload["teams"]}
    book = payload["records"]
    assert book["best"] and {(r["k"], r["s"]) for r in book["best"]} >= {("game", "pts"), ("season", "ppg"), ("career", "games"), ("team_season", "wins")}
    for r in book["best"] + book["log"]:
        assert (r["p"] in teams) if r["k"].startswith("team") else (str(r["p"]) in players)
    for r in book["log"]:
        assert (r["op"] in teams) if r["k"].startswith("team") else (str(r["op"]) in players)
        assert r["oy"] <= r["y"] and (r["v"] < r["o"] if r["d"] == "min" else r["v"] > r["o"])


def test_rivalry_history_covers_every_season(payload):
    rv, n = payload["rivalry"], payload["meta"]["years"]
    assert len(rv["years"]) == len(rv["level"]) == n and rv["pairs"]
    assert all(len(v) == n for v in rv["hist"].values())
    for p in rv["pairs"]:
        assert str(p["a"]) in payload["houses"] and str(p["b"]) in payload["houses"] and p["a"] < p["b"]
        assert f"{p['a']}-{p['b']}" in rv["hist"]
    for pk in rv["top"].values():
        assert len(pk) <= 3


def test_hall_of_fame_and_player_totals(payload):
    players = payload["players"]
    for h in payload["hof"]:
        assert players[str(h["p"])]["hof"] == h["y"]
    assert all(str(b["p"]) in players and str(b["h"]) in payload["houses"] for b in payload["builders"])
    with_career = [p for p in players.values() if "s" in p]
    assert with_career and all(len(p["tot"]) == 7 and p["tot"][5] == sum(r[3] for r in p["s"]) for p in with_career)
    assert all(abs(sum(p["an"]) - 1) < 0.02 for p in players.values() if p["hy"])


def test_hall_of_fame_rows_carry_the_career_role(payload):
    assert all(h["role"] in (None, "guard", "forward", "center") for h in payload["hof"])
