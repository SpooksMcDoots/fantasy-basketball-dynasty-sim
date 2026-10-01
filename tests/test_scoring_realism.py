"""Scoring realism: diminishing returns on shooting skill, a usage/efficiency trade-off, foul drawing from quickness and
handling, and intentional fouling of poor free-throw shooters."""
import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.core.types import TRAIT_IDX
from dynasty_sim.engine import kernel
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players
from dynasty_sim.engine.interface import NumbaEngine
from dynasty_sim.engine.params import load_engine_params
from dynasty_sim.league.pool import selected_units

REF, P = frozen_reference(), load_params()
PRM = load_engine_params(REF, {"touch_sat": 2.5, "usage_cost": 0.10, "usage_cost_hi": 1.0, "usage_hi": 2.0, "sf_drive": 0.15, "z_sat": 2.0})


def shot(k, z_touch=0.0, usage_rel=1.0, drive=0.0, dstr=0.0, prm=PRM):
    return kernel._shot(k, prm, 0.0, 0.0, z_touch, dstr, 0.0, 0.0, 235.0, False, 1.0, drive, usage_rel)


# ---- diminishing returns on shooting skill -------------------------------------------------------------------
@pytest.mark.parametrize("k", [0, 1, 2])
def test_shooting_skill_helps_but_saturates(k):
    pm = [shot(k, z)[0] for z in (0.0, 1.0, 2.5, 4.0, 8.0)]
    assert pm[0] < pm[1] < pm[2] < pm[3] <= pm[4]
    assert pm[4] - pm[2] < 0.5 * (pm[2] - pm[0])            # past the cap, extra skill adds little
    assert pm[4] - pm[3] < 0.15 * (pm[2] - pm[0])           # beyond 4 SD, doubling the skill adds <15% of what the first 2.5 SD did


# ---- efficiency falls as usage rises -----------------------------------------------------------------------------
def test_heavy_usage_costs_efficiency_but_light_usage_is_not_rewarded():
    even, heavy, light = shot(2, 1.0, usage_rel=1.0)[0], shot(2, 1.0, usage_rel=2.0)[0], shot(2, 1.0, usage_rel=0.4)[0]
    assert heavy < even
    assert light == pytest.approx(even)


# ---- foul drawing ---------------------------------------------------------------------------------------------------
def test_handling_draws_fouls_on_drives_but_not_on_threes():
    for k in (0, 1):
        assert shot(k, drive=1.5)[2] > shot(k, drive=0.0)[2] > shot(k, drive=-1.5)[2]
    assert shot(2, drive=1.5)[2] == pytest.approx(shot(2, drive=-1.5)[2])


def test_strength_still_draws_fouls_but_saturates():
    a, b, c = (shot(0, dstr=x)[2] for x in (0.0, 2.0, 6.0))
    assert a < b <= c and c - b < 0.5 * (b - a)


# ---- intentional fouling ("hack-a-Shaq") ------------------------------------------------------------------------------
def _team(rng, bad_shooter: bool):
    units = selected_units(13, rng, P, REF).copy()
    units[:, TRAIT_IDX["touch"]] = np.maximum(units[:, TRAIT_IDX["touch"]], 55.0)   # every teammate can shoot free throws
    if bad_shooter:                                         # tall enough to stay on the floor, but cannot shoot free throws
        units[0, TRAIT_IDX["height"]] += 18
        units[0, TRAIT_IDX["touch"]] = 12.0
    rows = build_players(units, REF)
    rows[0, S.C_OVR] = rows[:, S.C_OVR].max() + 1.0         # a starter, whatever his composite says
    return rows


def _run(rows_a, rows_b, prm_over, n=1500):
    eng = NumbaEngine(overrides=prm_over)
    PL = np.zeros((n, 2, 13, S.NF))
    PL[:, 0], PL[:, 1] = rows_a, rows_b
    co = np.zeros((n, 2, S.NCO))
    co[:, :, S.CO_DEPTH] = 10
    return eng.simulate_arrays(PL, np.full((n, 2), 13), co, np.arange(n) + 42)


def test_a_poor_free_throw_shooter_gets_fouled_more_when_the_other_team_can_hack():
    rng = np.random.default_rng(3)
    hacked_team, opponent = _team(rng, True), _team(rng, False)
    off = _run(hacked_team, opponent, {"hack_p": 0.0})
    on = _run(hacked_team, opponent, {"hack_p": 0.45})
    fta_off, fta_on = off[0][:, 0, 0, S.ST_FTA].mean(), on[0][:, 0, 0, S.ST_FTA].mean()
    assert off[0][:, 0, 0, S.ST_SEC].mean() > 600                                # he really does play
    assert fta_on > fta_off + 0.3, (fta_off, fta_on)                              # extra free throws from intentional fouls
    assert on[0][:, 1, :, S.ST_PF].sum() > off[0][:, 1, :, S.ST_PF].sum()         # the opponent commits the fouls


def test_hacking_needs_the_window_and_the_margin():
    rng = np.random.default_rng(3)
    a, b = _team(rng, True), _team(rng, False)
    base = _run(a, b, {"hack_p": 0.0}, n=400)
    for over in ({"hack_p": 0.45, "hack_secs": 0.0}, {"hack_p": 0.45, "hack_margin": -1.0}):
        got = _run(a, b, over, n=400)
        assert np.array_equal(got[0], base[0]) and np.array_equal(got[1], base[1])      # never fires: identical games


def test_no_hack_when_the_shooter_is_decent():
    rng = np.random.default_rng(4)
    a, b = _team(rng, False), _team(rng, False)
    off = _run(a, b, {"hack_p": 0.0}, n=600)
    on = _run(a, b, {"hack_p": 0.45}, n=600)
    assert np.array_equal(off[0], on[0])                    # nobody on the floor shoots below the threshold: identical


# ---- a volume ceiling: the cost steepens only for a one-man offence -------------------------------------------------
def test_usage_cost_steepens_only_above_the_ceiling():
    base = shot(2, 1.0, usage_rel=1.0)[0]
    below = [base - shot(2, 1.0, usage_rel=u)[0] for u in (1.5, 1.8, 2.0)]
    above = [base - shot(2, 1.0, usage_rel=u)[0] for u in (2.0, 2.1, 2.2)]
    assert all(b > 0 for b in below)
    step_below = (below[2] - below[1]) / 0.2             # cost per unit of share just under the ceiling
    step_above = (above[2] - above[1]) / 0.1             # and just over it
    assert step_above > 2 * step_below
    off = load_engine_params(REF, {"touch_sat": 2.5, "usage_cost": 0.10, "usage_cost_hi": 0.0, "usage_hi": 2.0, "z_sat": 2.0})
    assert shot(2, 1.0, usage_rel=2.2, prm=off)[0] > shot(2, 1.0, usage_rel=2.2)[0]


# ---- size earns touches --------------------------------------------------------------------------------------------------
def test_a_giant_with_poor_touch_takes_more_shots_when_size_counts_toward_usage():
    rng = np.random.default_rng(5)
    units = selected_units(13, rng, P, REF).copy()
    units[:, TRAIT_IDX["touch"]] = np.maximum(units[:, TRAIT_IDX["touch"]], 55.0)
    units[0] = P.race_mu[1]                                  # a typical Goliath: huge and strong, with poor touch
    rows = build_players(units, REF)
    rows[0, S.C_OVR] = rows[:, S.C_OVR].max() + 1.0
    other = _team(np.random.default_rng(6), False)
    none = _run(rows, other, {"hack_p": 0.0, "usage_size": 0.0}, n=500)
    some = _run(rows, other, {"hack_p": 0.0, "usage_size": 0.8}, n=500)
    fga = lambda r: r[0][:, 0, 0, S.ST_FGA].mean()
    assert r_sec(some) > 600 and fga(some) > 1.5 * fga(none)
    assert some[0][:, 0, 1:, S.ST_FGA].sum() < none[0][:, 0, 1:, S.ST_FGA].sum()          # the shots come from teammates


def r_sec(r):
    return r[0][:, 0, 0, S.ST_SEC].mean()
