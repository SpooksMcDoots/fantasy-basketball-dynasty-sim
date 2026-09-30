import re
import time
from pathlib import Path

import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players
from dynasty_sim.engine.interface import CoachParams, GameInput, NumbaEngine, TeamInput, _pack
from dynasty_sim.league.pool import deal_teams, selected_players

P, REF = load_params(), frozen_reference()


@pytest.fixture(scope="module")
def teams():
    return deal_teams(selected_players(8 * 13, np.random.default_rng(1), P, REF), 8)


@pytest.fixture(scope="module")
def eng():
    return NumbaEngine()


def _fixtures(teams, n):
    pairs = [(i % 8, (i % 8 + 1 + (i // 8) % 7) % 8) for i in range(n)]
    return [GameInput(TeamInput(teams[a]), TeamInput(teams[b])) for a, b in pairs]


@pytest.fixture(scope="module")
def batch(eng, teams):
    gis = _fixtures(teams, 1000)
    PL, NPL, CO = _pack(gis)
    eng.simulate_arrays(PL[:8], NPL[:8], CO[:8], np.arange(8))       # warm / compile
    t = time.perf_counter()
    out = eng.simulate_arrays(PL, NPL, CO, np.arange(1000))
    return out, time.perf_counter() - t, (PL, NPL, CO)


def test_1000_games_under_one_second(batch):
    assert batch[1] < 1.0, batch[1]


def test_no_nan_and_no_ties(batch):
    (BOX, SCORE, POSS), _, _ = batch
    assert np.isfinite(BOX).all() and (BOX >= 0).all()
    assert (SCORE[:, 0] != SCORE[:, 1]).all()


def test_box_sums_are_consistent(batch):
    (BOX, SCORE, POSS), _, _ = batch
    tm = BOX.sum(2)                                           # [G, 2, NSTAT]
    col = lambda s: tm[:, :, s]
    pts = 2 * (col(S.ST_FGM) - col(S.ST_TPM)) + 3 * col(S.ST_TPM) + col(S.ST_FTM)
    assert np.array_equal(pts, SCORE) and np.array_equal(col(S.ST_PTS), SCORE)
    assert (col(S.ST_FGM) <= col(S.ST_FGA)).all() and (col(S.ST_TPM) <= col(S.ST_TPA)).all()
    assert (col(S.ST_TPA) <= col(S.ST_FGA)).all() and (col(S.ST_FTM) <= col(S.ST_FTA)).all()
    assert (col(S.ST_AST) <= col(S.ST_FGM)).all()
    # five players on court at all times: minutes are 5 * elapsed, elapsed = 2880 + 300 * n_overtimes
    ot = (col(S.ST_SEC)[:, 0] / 5 - 2880) / 300
    assert np.allclose(col(S.ST_SEC)[:, 0], col(S.ST_SEC)[:, 1]) and np.allclose(ot, np.round(ot)) and (ot > -1e-9).all()
    assert (ot > 0.5).any(), "no overtime games in 1000 - overtime path untested"
    # every defensive rebound / steal / block has a matching opponent event
    misses = lambda t: (col(S.ST_FGA) - col(S.ST_FGM))[:, t] + (col(S.ST_FTA) - col(S.ST_FTM))[:, t]
    for t in (0, 1):
        assert (col(S.ST_ORB)[:, t] + col(S.ST_DRB)[:, 1 - t] <= misses(t)).all()
        assert (col(S.ST_BLK)[:, 1 - t] <= col(S.ST_FGA)[:, t]).all()
        assert (col(S.ST_STL)[:, 1 - t] <= col(S.ST_TOV)[:, t]).all()


def test_seed_determinism_and_batch_matches_single(eng, teams, batch):
    (BOX, SCORE, POSS), _, (PL, NPL, CO) = batch
    b2 = eng.simulate_arrays(PL[:50], NPL[:50], CO[:50], np.arange(50))
    assert np.array_equal(b2[0], BOX[:50]) and np.array_equal(b2[1], SCORE[:50])
    gis = _fixtures(teams, 3)
    for i, gi in enumerate(gis):
        r = eng.simulate(gi, i)
        assert np.array_equal(r.box, BOX[i]) and np.array_equal(r.score, SCORE[i]) and np.array_equal(r.poss, POSS[i])
    assert not np.array_equal(eng.simulate(gis[0], 1).box, eng.simulate(gis[0], 2).box)


def test_possession_log(eng, teams):
    r = eng.simulate(_fixtures(teams, 1)[0], 9, want_log=True)
    assert r.log is not None and 100 < len(r.log) <= 600
    assert set(np.unique(r.log[:, 3]).astype(int)) <= {1, 2, 3, 4, 5, 6}
    fg_pts = r.log[r.log[:, 3] == 4, 6].sum()
    assert fg_pts == 2 * (r.box[:, :, S.ST_FGM].sum() - r.box[:, :, S.ST_TPM].sum()) + 3 * r.box[:, :, S.ST_TPM].sum()
    assert r.log[:, 1].min() >= 0 and r.log[:, 0].max() <= 12


def test_unavailable_players_never_play(eng, teams):
    home = teams[0].copy()
    home[:3, S.C_AVAIL] = 0.0
    r = eng.simulate(GameInput(TeamInput(home), TeamInput(teams[1])), 4)
    assert (r.box[0, :3] == 0).all() and r.box[0, 3:, S.ST_SEC].sum() > 0


def test_needs_five_available(eng, teams):
    bad = teams[0].copy()
    bad[4:, S.C_AVAIL] = 0.0
    with pytest.raises(ValueError):
        eng.simulate(GameInput(TeamInput(bad), TeamInput(teams[1])), 1)


def test_rotation_depth_limits_who_plays(eng, teams):
    r = eng.simulate(GameInput(TeamInput(teams[0], CoachParams(rotation_depth=6)), TeamInput(teams[1])), 3)
    assert (r.box[0, 6:, S.ST_SEC] == 0).all()


# ---- direction-of-effect checks: physics, not labels -----------------------------------
def _match(eng, a, b, n=600):
    gis = [GameInput(TeamInput(a), TeamInput(b)) for _ in range(n)]
    PL, NPL, CO = _pack(gis)
    BOX, SCORE, POSS = eng.simulate_arrays(PL, NPL, CO, np.arange(n) + 5000)
    return BOX.sum(2), SCORE


def _shifted(teams, trait, delta):
    from dynasty_sim.core.types import TRAIT_IDX
    base = teams[0]
    units = np.tile(P.race_mu[0], (13, 1)).copy()
    units[:] = P.race_mu[0]
    units[:, TRAIT_IDX[trait]] += delta
    return build_players(units, REF)


@pytest.mark.parametrize("trait,delta,stat,sign", [
    ("height", 20.0, S.ST_BLK, +1),        # taller team blocks more
    ("height", 20.0, "orb_rate", +1),      # ...and wins more of the offensive-rebound contests
    ("touch", 12.0, S.ST_FGM, +1),         # better shooters make more
    ("agility", -0.6, S.ST_STL, +1),       # faster (fewer seconds) team steals more
])
def test_trait_moves_stat_in_expected_direction(eng, teams, trait, delta, stat, sign):
    a, b = _shifted(teams, trait, delta), _shifted(teams, trait, 0.0)
    ta, _ = _match(eng, a, b)
    if stat == "orb_rate":       # raw ORB also depends on how often each team misses
        rate = lambda t: ta[:, t, S.ST_ORB].sum() / (ta[:, t, S.ST_ORB].sum() + ta[:, 1 - t, S.ST_DRB].sum())
        assert sign * (rate(0) - rate(1)) > 0
    else:
        assert sign * (ta[:, 0, stat].mean() - ta[:, 1, stat].mean()) > 0


def test_identical_teams_have_home_edge(eng, teams):
    t, score = _match(eng, teams[0], teams[0], n=2000)
    assert (score[:, 0] - score[:, 1]).mean() > 0


# ---- structural guard: the engine never sees group labels ----------------------------------
def test_engine_source_has_no_group_labels():
    banned = re.compile(r"\b(race|races|ancestry|goliath|elf|human)\b", re.I)
    for f in Path(__file__).parent.parent.joinpath("dynasty_sim", "engine").glob("*.py"):
        hits = banned.findall(f.read_text())
        assert not hits, (f.name, hits)
    assert not any(re.search("race|ancest", n) for n in ("C_HEIGHT", *dir(S)) if n.isupper())
