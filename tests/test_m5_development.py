import numpy as np
import pytest

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.core.types import Person, RACE_IDX, TRAIT_IDX
from dynasty_sim.development import aging, injury, learning
from dynasty_sim.development.growth import develop_year, new_state
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players
from dynasty_sim.league.pool import selected_founders

P, REF = load_params(), frozen_reference()


def _careers(race: str, n: int, seed: int, years: int):
    """ovr trajectories [n, years] for n selected players followed from the adult age."""
    rng = np.random.default_rng(seed)
    r = RACE_IDX[race]
    a0 = int(P.adult_age[r])
    units = np.zeros((n, years, 12))
    ath = np.zeros((n, years, 4))
    skill = np.zeros((n, years, 3))
    for i, g in enumerate(selected_founders(n, race, rng, P, REF)):
        st = new_state(Person(i, "M", 0, g), a0, P, rng, REF)
        for t in range(years):
            units[i, t], ath[i, t], skill[i, t] = st.units(), st.ath_dev, st.skill_z
            develop_year(st, a0 + t, P, rng)
    ovr = build_players(units.reshape(-1, 12), REF)[:, S.C_OVR].reshape(n, years)
    return a0, ovr, ath, skill


def test_human_mean_peak_age_in_target():
    a0, ovr, _, _ = _careers("human", 500, 3, 22)
    peak = a0 + ovr.argmax(1)
    assert 25.5 <= peak.mean() <= 28.0, peak.mean()


def test_other_races_peak_at_their_own_ages():
    a0, ovr, _, _ = _careers("goliath", 300, 4, 22)
    assert 26.0 <= (a0 + ovr.argmax(1)).mean() <= 29.5                 # spec: 26-29
    a0, ovr, _, _ = _careers("elf", 300, 5, 44)
    assert 32.0 <= (a0 + ovr.argmax(1)).mean() <= 48.0                 # spec: 32-48


def test_athletic_growth_then_decline_and_skill_growth():
    a0, ovr, ath, skill = _careers("human", 300, 6, 20)
    m = ath.mean(0)                                                     # [years, 4]
    assert (m[6] > m[0]).all()                                          # grows through the mid-20s
    assert (m[16] < m[6]).all()                                         # then declines
    assert (skill.mean(0)[6, :2] > skill.mean(0)[0, :2]).all()          # touch and vision improve
    assert skill.mean(0)[19, 2] >= skill.mean(0)[12, 2] - 0.05          # temperament does not erode


def test_age_eff_and_race_scaling():
    assert aging.age_eff(20, RACE_IDX["human"], P) == 20
    assert aging.age_eff(24, RACE_IDX["goliath"], P) == 20
    assert aging.age_eff(30, RACE_IDX["elf"], P) == 20
    assert aging.age_eff(40, RACE_IDX["elf"], P) == pytest.approx(25.0)
    assert aging.age_eff(35, RACE_IDX["goliath"], P) > 35                # goliaths age faster than the calendar


def test_decline_shape_follows_zengm_points():
    inc = aging.increment
    assert inc("agility", 29) == pytest.approx(-0.2) and inc("agility", 33) == pytest.approx(-0.3)
    assert inc("vertical", 28) == pytest.approx(-0.3) and inc("vertical", 26) == 0.0
    assert inc("endurance", 30) == 0.0 and inc("endurance", 32) == pytest.approx(-0.2)
    assert inc("strength", 20) == pytest.approx(0.2) and inc("strength", 24) == pytest.approx(0.1)


def test_new_prospects_start_with_expected_growth_still_ahead():
    rng = np.random.default_rng(0)
    g = selected_founders(1, "human", rng, P, REF)[0]
    st = new_state(Person(1, "M", 0, g), 20, P, rng, REF)
    assert st.ath_dev.mean() == pytest.approx(-0.8, abs=0.15)
    assert (st.skill_z < st.skill_target).all()


def test_learning_window_multiplies_growth():
    assert learning.window_mult(15, RACE_IDX["human"], P) == 2.0
    assert learning.window_mult(25, RACE_IDX["human"], P) == 1.0
    assert learning.window_mult(30, RACE_IDX["elf"], P) == 2.2


# ---- injuries ------------------------------------------------------------------------------
def test_hazard_scales_with_minutes_and_multiplier():
    lo = injury.hazard(np.array([0.0, 36.0]), np.array([1.0, 1.0]), P)
    assert lo[1] > lo[0] > 0
    assert injury.hazard(np.array([20.0]), np.array([2.0]), P)[0] == pytest.approx(
        2 * injury.hazard(np.array([20.0]), np.array([1.0]), P)[0])


def test_severity_distribution():
    rng = np.random.default_rng(1)
    n = 60_000
    hit, types, games = injury.draw_injuries(np.ones(n), np.full(n, 25.0), rng, P)
    assert hit.all() and abs((games > 0).mean() - P.injury["missed_frac"]) < 0.01
    assert 2 <= np.median(games[games > 0]) <= 4
    assert games[types == 1][games[types == 1] > 0].mean() > games[types == 0][games[types == 0] > 0].mean()
    _, _, old = injury.draw_injuries(np.ones(n), np.full(n, 35.0), rng, P)
    assert old[old > 0].mean() > games[games > 0].mean()


def test_extreme_height_and_low_resilience_raise_hazard():
    rng = np.random.default_rng(2)
    hs = {}
    for name, dz in (("normal", 0.0), ("giant", 3.5)):
        g = selected_founders(1, "human", rng, P, REF)[0]
        g.A[:], g.E_perm[:], g.loci[:] = 0.0, 0.0, 0
        g.A[TRAIT_IDX["height"]] = dz
        hs[name] = new_state(Person(1, "M", 0, g), 22, P, rng, REF).hazard_mult
    assert hs["giant"] > 1.5 * hs["normal"]
    g = selected_founders(1, "human", rng, P, REF)[0]
    g.A[:], g.E_perm[:], g.loci[:] = 0.0, 0.0, 0
    g.loci[2] = 1                                                        # brittle homozygote: injury_res -2 SD
    assert new_state(Person(1, "M", 0, g), 22, P, rng, REF).hazard_mult > hs["normal"] * 1.8
    g.loci[:], g.loci[0] = 0, 1                                          # titan homozygote: 3x hazard
    assert new_state(Person(1, "M", 0, g), 22, P, rng, REF).hazard_mult > 2.5
