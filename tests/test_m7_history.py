import numpy as np
import pytest

from dynasty_sim.engine import schema as S
from dynasty_sim.history import awards as AW
from dynasty_sim.history import events as EV
from dynasty_sim.history.arcs import build_arcs, linked
from dynasty_sim.history.hof import DYNASTY_TITLES, HallOfFame
from dynasty_sim.history.names import NameBook
from dynasty_sim.history.records import RecordBook
from dynasty_sim.history.rivalry import CLOSE_MARGIN, DECAY, MIN_LEVEL, SUSTAINED, RivalryBoard


# ---- names -----------------------------------------------------------------------------------------
class _P:
    def __init__(self, race=0, mother=None, father=None):
        self.race, self.mother_id, self.father_id = race, mother, father


def _book(people, seed=1):
    return NameBook(seed, lambda pid: people.get(pid))


def test_names_are_deterministic_and_order_independent():
    people = {1: _P(), 2: _P(1), 3: _P(0, 1, 2)}
    a, b = _book(people), _book(people)
    forward = [a.person(i) for i in (1, 2, 3)]
    backward = [b.person(i) for i in (3, 2, 1)][::-1]
    assert forward == backward
    assert _book(people, seed=2).person(1) != a.person(1)


def test_children_inherit_the_fathers_surname_else_mothers():
    people = {1: _P(), 2: _P(), 3: _P(0, 1, 2), 4: _P(0, 1, None)}
    nb = _book(people)
    assert nb.surname(3) == nb.surname(2) and nb.surname(4) == nb.surname(1)
    assert " " in nb.person(3) and nb.person(3).split()[0] == nb.given(3)


def test_names_never_read_as_code_values(monkeypatch):
    nb = _book({}, seed=7)
    rng = lambda i: nb._rng(0, i)
    import dynasty_sim.history.names as N
    words = lambda: {nb._word(rng(i), "human", 2) for i in range(250_000)}
    monkeypatch.setattr(N, "_RESERVED", set())
    assert "None" in words()                                   # the syllable pools can spell it...
    monkeypatch.undo()
    assert not (words() & N._RESERVED)                         # ...and the generator redraws it


def test_team_and_house_names_are_distinct():
    nb = _book({})
    assert len({nb.team(i) for i in range(8)}) == 8
    assert len({nb.house(i) for i in range(1, 9)}) >= 7 and nb.house(1).startswith("House ")


# ---- events and arcs ---------------------------------------------------------------------------------------
def test_event_context_chains_previous_events_of_each_protagonist():
    log = EV.EventLog()
    a = log.add("award", 1, (7,), (1,), EV.NOTABLE)
    b = log.add("record", 2, (7,), (1,), EV.NOTABLE)
    c = log.add("title", 3, (8,), (1,), EV.MAJOR)
    m = log.add("retire", 4, (7,), (2,), EV.MINOR)              # minor events never enter chains
    assert a.context == () and b.context == (a.id,)
    assert c.context == (b.id,)                                   # shares house 1 with b
    assert m.context == ()
    d = log.add("hof", 5, (7,), (2,), EV.LANDMARK)
    assert d.context == (b.id,)                                   # the minor retirement did not enter player 7's chain


def test_arcs_need_three_notable_events_and_are_linked():
    log = EV.EventLog()
    for y in (1, 3, 5):
        log.add("award", y, (7,), (), EV.NOTABLE)
    log.add("award", 2, (9,), (), EV.NOTABLE)
    log.add("retire", 6, (7,), (), EV.MINOR)
    arcs = build_arcs(log.events)
    assert [a.protagonist for a in arcs] == [("person", 7)] and len(arcs[0].events) == 3
    assert linked(arcs[0])
    assert arcs[0].span == (1, 5)


def test_percentile_and_series_probabilities():
    assert EV.percentile_rank(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 0.5
    assert EV.percentile_rank(0, []) == 0.5
    assert EV.series_win_prob(0.5, 5) == pytest.approx(0.5)
    assert EV.series_win_prob(0.7, 5) > 0.83 and EV.series_win_prob(0.3, 5) < 0.17
    assert EV.game_win_prob(5.0, 5.0, home_edge=0.0) == pytest.approx(0.5)
    assert EV.game_win_prob(10, 0) > EV.game_win_prob(5, 0) > 0.5


# ---- records -------------------------------------------------------------------------------------------------
def test_records_only_report_genuine_breaks_once_per_season():
    book = RecordBook()
    book.offer("game", "pts", 30, holder=1, year=1, scopes=("all", "human"))
    book.offer("game", "pts", 41, holder=2, year=1, scopes=("all", "human"))       # improved within season 1
    assert book.finalize(1) == []                                                  # baseline: nothing to break
    assert book.holder("game", "pts").value == 41 and book.holder("game", "pts").holder == 2
    book.offer("game", "pts", 35, holder=3, year=2)
    assert book.finalize(2) == []                                                  # did not beat 41
    for v, h in ((44, 4), (52, 5)):
        book.offer("game", "pts", v, holder=h, year=3, scopes=("all",))
    brk = book.finalize(3)
    assert len(brk) == 1 and brk[0].new.value == 52 and brk[0].old.value == 41 and brk[0].old.holder == 2
    assert book.holder("game", "pts").holder == 5


def test_minima_and_scopes_are_tracked_separately():
    book = RecordBook()
    book.offer("team_game", "team_pts", 80, 1, 1, direction="min")
    book.offer("team_game", "team_pts", 120, 1, 1, direction="max")
    book.finalize(1)
    book.offer("team_game", "team_pts", 70, 2, 2, direction="min")
    book.offer("team_game", "team_pts", 110, 2, 2, direction="max")
    brk = book.finalize(2)
    assert [(b.direction, b.new.value) for b in brk] == [("min", 70.0)]
    book.offer("game", "reb", 20, 1, 3, scopes=("all", "elf"))
    book.finalize(3)
    book.offer("game", "reb", 22, 2, 4, scopes=("elf",))
    assert [b.scope for b in book.finalize(4)] == ["elf"]


# ---- awards ------------------------------------------------------------------------------------------------------
def _line(games=70, minutes=2400, **stats):
    v = np.zeros(S.NSTAT + 1)
    v[0] = games
    v[1 + S.ST_SEC] = minutes * 60
    for k, x in stats.items():
        v[1 + getattr(S, "ST_" + k.upper())] = x
    return v


def _stats(**stats):
    return _line(**stats)[1:]


def test_value_rewards_efficient_scoring_rebounds_and_punishes_turnovers():
    base = AW.raw_value(_stats(pts=1000, fgm=400, fga=900, ftm=150, fta=190))
    assert AW.raw_value(_stats(pts=1000, fgm=400, fga=800, ftm=150, fta=190)) > base       # same points, fewer shots
    assert AW.raw_value(_stats(pts=1000, fgm=400, fga=900, ftm=150, fta=190, drb=500)) > base
    assert AW.raw_value(_stats(pts=1000, fgm=400, fga=900, ftm=150, fta=190, tov=300)) < base
    assert AW.raw_value(_stats(pts=1000, fgm=400, fga=900, ftm=150, fta=190, stl=100, blk=80)) > base
    assert AW.W_SHOOT + AW.W_TOV + AW.W_REB + AW.W_FT == pytest.approx(1.0)


def test_awards_pick_expected_winners_and_respect_qualification():
    lines = {1: _line(pts=2300, fgm=850, fga=1650, ftm=450, fta=520, ast=450, orb=150, drb=550, stl=70),   # star scorer
             2: _line(pts=700, fgm=270, fga=600, ftm=100, fta=130, stl=100, blk=100, drb=450, orb=100),    # defender
             3: _line(games=10, minutes=200, pts=900, fgm=350, fga=500, ftm=100, fta=110),               # too few games
             4: _line(pts=1200, fgm=450, fga=1000, ftm=120, fta=150, ast=300, orb=80, drb=250)}
    values = AW.season_values(lines)
    aw = AW.yearly_awards(lines, values, {1: 0, 2: 1, 3: 0, 4: 1}, rookies={2, 4}, prev_values={1: 10.0, 4: -50.0}, champion_team=1)
    assert aw["mvp"] == 1 and aw["scoring"] == 1 and aw["dpoy"] == 2
    assert 3 not in aw["all_league"] and aw["all_league"][0] == 1 and len(aw["all_league"]) == 3
    assert aw["roy"] == max((2, 4), key=lambda p: values[p])
    assert aw["most_improved"] == max((1, 4), key=lambda p: values[p] - {1: 10.0, 4: -50.0}[p])
    assert aw["finals_mvp"] == max((2, 4), key=lambda p: values[p])


def test_finals_mvp_is_judged_on_the_finals():
    lines = {1: _line(pts=2300, fgm=850, fga=1650, ftm=450, fta=520, ast=450, orb=150, drb=550, stl=70),   # season star
             2: _line(pts=700, fgm=270, fga=600, ftm=100, fta=130, stl=100, blk=100, drb=450, orb=100)}
    values = AW.season_values(lines)
    finals = {1: _line(games=4, minutes=140, pts=60, fgm=22, fga=70, ftm=10, fta=14, tov=12),           # cold series
              2: _line(games=4, minutes=140, pts=110, fgm=42, fga=70, ftm=20, fta=24, drb=40, orb=12)}   # hot series
    aw = AW.yearly_awards(lines, values, {1: 0, 2: 0}, rookies=set(), prev_values={}, champion_team=0, finals=finals)
    assert aw["finals_mvp"] == 2
    assert AW.yearly_awards(lines, values, {1: 0, 2: 0}, set(), {}, 0)["finals_mvp"] == 1    # no finals data: season value
    assert AW.finals_line_text(finals[2]) == "27.5 pts, 13.0 reb, 0.0 ast in 4 finals games"


def test_mvp_blends_value_scoring_and_team_success():
    big = _line(pts=500, fgm=250, fga=330, orb=300, drb=700, stl=100, blk=200)           # top value, 7 ppg
    scorer = _line(pts=2400, fgm=900, fga=1750, ftm=450, fta=520, ast=300, orb=100, drb=400)   # 34 ppg
    allround = _line(pts=1960, fgm=760, fga=1500, ftm=300, fta=360, ast=500, orb=100, drb=500, stl=90)   # 28 ppg
    filler = _line(pts=700, fgm=280, fga=650, ftm=70, fta=90, ast=100, orb=60, drb=200)
    lines = {1: big, 2: scorer, 3: allround, 4: filler}
    values = {1: 300.0, 2: 270.0, 3: 285.0, 4: 40.0}
    team_of = {1: 0, 2: 1, 3: 2, 4: 3}
    pick = lambda win: AW.mvp_pick([1, 2, 3, 4], lines, values, team_of, win)
    assert pick(None) != 1                                       # best value alone no longer wins with 7 points a game
    assert pick({0: 0.5, 1: 0.35, 2: 0.80, 3: 0.4}) == 3         # a near-equal star on the best team beats a scorer on a loser
    assert pick({0: 0.5, 1: 0.80, 2: 0.35, 3: 0.4}) == 2         # and the other way round
    assert sum(AW.MVP_WEIGHTS) == pytest.approx(1.0)


def test_all_league_is_a_lineup_when_roles_are_known():
    values = {p: 100.0 - p for p in range(1, 13)}                          # player 1 is the best, 12 the worst
    roles = {p: ("center" if p <= 4 else "forward" if p <= 8 else "guard") for p in values}
    team = AW.all_league_team(list(values), values, roles)
    assert sorted(roles[p] for p in team) == ["center", "forward", "forward", "guard", "guard"]
    assert team[0] == 1 and team == sorted(team)                           # best value first
    assert AW.all_league_team(list(values), values, None) == [1, 2, 3, 4, 5]
    assert len(AW.all_league_team([1, 2, 3], values, {1: "guard", 2: "guard", 3: "guard"})) == 3


def test_roles_split_the_league_into_height_thirds():
    r = AW.assign_roles({p: 170.0 + p for p in range(9)})
    assert [r[p] for p in range(9)] == ["guard"] * 3 + ["forward"] * 3 + ["center"] * 3


# ---- hall of fame ---------------------------------------------------------------------------------------------------
class _S:
    def __init__(self, value):
        self.value = value


class _C:
    """A minimal career: per-season values, honours and a retirement year."""

    def __init__(self, pid, values, retired=10, awards=()):
        self.pid, self.seasons, self.retired_year, self.awards = pid, [_S(v) for v in values], retired, list(awards)

    def career_value(self):
        return float(sum(s.value for s in self.seasons))


def test_hall_of_fame_paces_inductions_and_prefers_the_best():
    hof = HallOfFame()
    careers = {i: _C(i, [float(i)] * 8) for i in range(1, 101)}       # 100 retired careers; quality rises with id
    inducted = []
    for year in range(20, 40):
        inducted += hof.inductees_this_year(year, careers)
    assert 23 <= len(inducted) <= 27                                    # about 1.25 per season over 20 seasons
    assert inducted[0] == 100 and min(inducted) >= 70                   # only above the 70th-percentile floor, best first
    assert len(set(inducted)) == len(inducted)


def test_a_long_mediocre_career_loses_to_a_short_great_one():
    """Longevity alone cannot carry a player: 25 average seasons must rank below 8 elite ones."""
    from dynasty_sim.history.hof import peak_value, score_careers
    filler = {i: _C(i, [40.0 + i] * 9, awards=[(1, "All-League")] * (i % 4)) for i in range(1, 40)}   # a realistic pool: many strong peaks
    long_average = _C(100, [22.0] * 25)                                  # career 550 (mid-pack), best five 110 (bottom)
    short_great = _C(101, [70.0] * 8)                                    # career 560, best five 350
    scores = score_careers(list(filler.values()) + [long_average, short_great], {})
    assert long_average.career_value() < 1.05 * short_great.career_value()
    assert peak_value(short_great) > 3 * peak_value(long_average)
    assert scores[101]["score"] > scores[100]["score"]
    for_lack_of_peak = HallOfFame()
    pool = {**filler, 100: long_average}
    for year in range(20, 60):
        for_lack_of_peak.inductees_this_year(year, pool)
    assert len(for_lack_of_peak.players) >= 10 and 100 not in for_lack_of_peak.players   # never inducted on length alone


def test_honours_and_titles_lift_a_career_of_equal_value():
    from dynasty_sim.history.hof import honour_points, score_careers
    a = _C(1, [30.0] * 8)
    b = _C(2, [30.0] * 8, awards=[(3, "MVP"), (4, "MVP"), (5, "All-League")])
    filler = [_C(10 + i, [10.0 + i] * 8) for i in range(30)]
    assert honour_points(b, {2: 2}) == 10 + 10 + 3 + 2 * 3 and honour_points(a, {}) == 0
    scores = score_careers([a, b] + filler, {2: 2})
    assert scores[2]["score"] > scores[1]["score"]


def test_each_career_is_judged_against_its_own_role():
    from dynasty_sim.history.hof import score_careers
    centers = [_C(i, [150.0 + 5 * i] * 8) for i in range(20)]              # centers rack up a lot of value
    guards = [_C(100 + i, [40.0 + 2 * i] * 8) for i in range(20)]           # guards rack up much less
    roles = {**{c.pid: "center" for c in centers}, **{c.pid: "guard" for c in guards}}
    pooled = score_careers(centers + guards, {})
    by_role = score_careers(centers + guards, {}, roles)
    assert min(pooled[c.pid]["score"] for c in centers) > max(pooled[g.pid]["score"] for g in guards)     # pooled: every center beats every guard
    assert by_role[119]["score"] == pytest.approx(by_role[19]["score"])                                   # best guard ties best center
    assert by_role[100]["role"] == "guard" and by_role[0]["role"] == "center"
    top = sorted(by_role, key=lambda p: -by_role[p]["score"])[:8]
    assert 3 <= sum(p >= 100 for p in top) <= 5                             # a role-balanced top of the class
    few_roles = {**{c.pid: "center" for c in centers}, **{g.pid: "guard" for g in guards[:3]}}
    few = score_careers(centers + guards[:3], {}, few_roles)
    assert few[100]["score"] == pytest.approx(score_careers(centers + guards[:3], {})[100]["score"])       # too few guards: pooled


def test_hall_of_fame_needs_eligibility_and_dynasty_builders():
    hof = HallOfFame()
    careers = {i: _C(i, [100.0 + i] * 3) for i in range(30)}             # too short a career
    careers.update({100 + i: _C(100 + i, [50.0 + i] * 9, retired=None) for i in range(30)})   # still playing
    assert hof.inductees_this_year(50, careers) == []
    got = hof.builders_this_year(50, [(1, 4, 0, 30, DYNASTY_TITLES), (2, 5, 0, 20, DYNASTY_TITLES - 1)])
    assert got == [(1, 4, DYNASTY_TITLES)] and hof.builders_this_year(51, [(1, 4, 0, 30, 9)]) == []


# ---- rivalry index -----------------------------------------------------------------------------------------------------
def test_rivalry_points_follow_the_spec():
    rv = RivalryBoard()
    rv.game(1, 2, 20)
    assert rv.index[(1, 2)] == 1.0
    rv.game(2, 1, -CLOSE_MARGIN)                          # close game: +1 +2
    assert rv.index[(1, 2)] == 4.0
    rv.playoff_series(1, 2)
    rv.marriage_or_dispute(2, 1, "marriages")
    rv.player_move(1, 2)
    assert rv.index[(1, 2)] == 4.0 + 5 + 8 + 4
    rv.add(3, 3, 50, "x")
    assert (3, 3) not in rv.index


def test_rivalry_flags_a_sustained_outlier_but_not_an_even_field():
    rv = RivalryBoard()
    for year in range(1, 13):
        for a in range(1, 6):
            for b in range(a + 1, 6):
                rv.add(a, b, 22.5, "games")                              # every pairing identical (steady state ~127)
        rv.add(1, 2, 6.75, "playoffs")                                   # one pairing runs ~30% hotter (steady ~166)
        rv.end_season(year)
    assert list(rv.flagged) == [(1, 2)]
    assert rv.streak[(1, 2)] >= SUSTAINED and rv.why((1, 2))[0] == "games"


def test_rivalry_index_decays_each_season():
    rv = RivalryBoard()
    rv.add(1, 2, 100.0, "games")
    rv.end_season(1)
    assert rv.index[(1, 2)] == pytest.approx(100.0 * DECAY) and rv.level >= MIN_LEVEL
    rv.end_season(2)
    assert rv.index[(1, 2)] == pytest.approx(100.0 * DECAY ** 2)
