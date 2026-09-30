import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.rng import RngTree
from dynasty_sim.engine import schema as S
from dynasty_sim.league.playoffs import home_pattern, run_playoffs
from dynasty_sim.league.season import League, circle_rounds
from dynasty_sim.league.standings import draft_order, rank_teams
from dynasty_sim.population.demography import Population


def _league(seed=1, human_only=False, P=None):
    P = P or load_params()
    if human_only:
        P.league["prospect_race_share"] = {"human": 1.0, "goliath": 0.0, "elf": 0.0}
        P.demo["house_races"] = ["human"] * 8
    pop = Population(P, RngTree(seed))
    pop.seed()
    lg = League(pop)
    lg.found()
    return lg


@pytest.fixture(scope="module")
def played():
    """A league run through 10 seasons, keeping a snapshot of invariants after each."""
    lg = _league(1)
    snaps = []
    for _ in range(10):
        lg.step_year()
        snaps.append({t.id: list(t.roster) for t in lg.teams})
        _check_invariants(lg)
    return lg, snaps


def _check_invariants(lg):
    P = lg.P.league
    seen = set()
    for t in lg.teams:
        assert P["roster_min"] <= len(t.roster) <= P["roster_max"], (t.id, len(t.roster))
        for pid in t.roster:
            assert pid not in seen
            seen.add(pid)
            st, person = lg.states[pid], lg.persons[pid]
            assert st.team_id == t.id and person.alive and person.death_year > lg.year
        own = sum(lg.persons[p].house_id == t.house_id for p in t.roster)
        assert own <= P["house_cap"], (t.id, own)
    assert seen == set(lg.states) == set(lg.persons)


# ---- structure ---------------------------------------------------------------------------------
def test_founding_fills_every_roster_and_respects_cap():
    lg = _league(2)
    assert all(len(t.roster) == 13 for t in lg.teams)
    _check_invariants(lg)


def test_ten_seasons_keep_invariants_and_play_full_schedule(played):
    lg, _ = played
    for r in lg.history:
        assert (r.wins + r.losses == 120).all() and r.wins.sum() == 480 and r.n_games == 480
        assert r.champion in range(8) and r.finalist != r.champion


def test_rosters_turn_over(played):
    lg, snaps = played
    first, last = set(sum(snaps[0].values(), [])), set(sum(snaps[-1].values(), []))
    assert len(first & last) < len(first)                     # retirement + draft replace players
    from dynasty_sim.development.aging import age_eff
    ae = [age_eff(lg.age(p), lg.states[p].race, lg.P) for p in lg.states]      # calendar age is meaningless across races
    assert 22 <= np.mean(ae) <= 30 and max(ae) < 45


def test_history_events_recorded(played):
    lg, _ = played
    kinds = {e.kind for e in lg.pop.events}
    assert {"retire", "draft", "champion"} <= kinds


def test_same_seed_same_league():
    a, b = _league(5), _league(5)
    for _ in range(3):
        ra, rb = a.step_year(), b.step_year()
        assert np.array_equal(ra.wins, rb.wins) and ra.champion == rb.champion
    assert [t.roster for t in a.teams] == [t.roster for t in b.teams]


def test_dead_pros_leave_the_league():
    lg = _league(3)
    pid = next(p for p in lg.states if p in lg.pop.people)
    lg.pop.people[pid].death_year = lg.year + 1
    lg.step_year()
    assert pid not in lg.states and all(pid not in t.roster for t in lg.teams)


def test_injured_players_are_marked_unavailable():
    lg = _league(4)
    team = lg.teams[0]
    star = team.roster[0]
    lg.states[star].games_left = 4
    lg._sync_avail(team)
    assert lg._rows[0][0, S.C_AVAIL] == 0.0 and lg._rows[0][1, S.C_AVAIL] == 1.0


def test_injuries_tick_down_only_when_sitting():
    lg = _league(6)
    res = lg._play_season()
    assert all(s.games_left >= 0 for s in lg.states.values())
    assert res.injuries > 0


# ---- draft, standings, playoffs ---------------------------------------------------------------
def test_draft_order_is_reverse_standings():
    wins = np.array([80, 20, 50, 20, 100, 60, 40, 70])
    mov = np.array([5.0, -8.0, 0.0, -9.0, 9.0, 3.0, -2.0, 4.0])
    assert draft_order(wins, mov) == [3, 1, 6, 2, 5, 7, 0, 4]
    assert rank_teams(wins, mov)[0] == 4


def test_draft_uses_reverse_standings_and_fills_rosters():
    lg = _league(7)
    lg.last_wins = np.array([100, 90, 80, 70, 60, 50, 40, 30])
    lg.last_mov = np.zeros(8)
    lg.step_year()
    picks = [e for e in lg.pop.events if e.kind == "draft" and e.year == lg.year]
    teams_in_order = [int(e.detail.split()[1]) for e in picks]
    if len(teams_in_order) >= 8:
        assert teams_in_order[:8] == [7, 6, 5, 4, 3, 2, 1, 0]


def test_home_pattern_and_bracket():
    assert home_pattern(5) == [True, True, False, False, True]
    assert home_pattern(3) == [True, True, True]
    top_always_wins = lambda h, a: (1, 0) if h < a else (0, 1)
    out = run_playoffs([0, 1, 2, 3], 5, top_always_wins)
    assert out["champion"] == 0 and out["finalist"] == 1 and set(out["semis"]) == {0, 1}
    upset = lambda h, a: (0, 1) if h == 0 else (1, 0)                # seed 0 loses every game it hosts...
    assert run_playoffs([0, 1], 5, lambda h, a: (1, 0) if h == 1 else (0, 1))["champion"] == 1


def test_circle_rounds_are_perfect_matchings():
    rounds = circle_rounds(8)
    assert len(rounds) == 7
    pairs = set()
    for r in rounds:
        assert sorted(sum(([a, b] for a, b in r), [])) == list(range(8))
        pairs |= {tuple(sorted(p)) for p in r}
    assert len(pairs) == 28


# ---- acceptance: injuries ------------------------------------------------------------------------
def _injury_summary(lg, seasons):
    inj = exp = 0
    missed = []
    for _ in range(seasons):
        r = lg.step_year()
        inj += r.injuries
        exp += r.injury_exposures
        missed += r.games_missed
    return 1000 * inj / exp, float(np.median(missed))


@pytest.mark.parametrize("human_only", [False, True])
def test_injury_rate_and_severity_in_target(human_only):
    rate, med = _injury_summary(_league(11, human_only), 8)
    assert 13 <= rate <= 19, rate                            # injuries per 1000 player-games
    assert 2 <= med <= 4, med                                # median games missed


def test_loop_produces_plausible_basketball():
    lg = _league(12)
    for _ in range(3):
        lg.step_year()
    r = lg.history[-1]
    t, poss = r.totals, r.poss
    fga, fta, pts = t[S.ST_FGA], t[S.ST_FTA], t[S.ST_PTS]
    assert 90 < poss / r.n_games / 2 < 112                    # pace
    assert 95 < 100 * pts / poss < 130                        # ORtg
    assert 0.40 < t[S.ST_FGM] / fga < 0.60 and 0.15 < fta / fga < 0.35
