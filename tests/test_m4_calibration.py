"""M4: the frozen calibration keeps a Human league inside every validation-table range."""
import numpy as np
import pytest
import yaml

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.engine.calibrate import (
    CALIBRATED, build_league_set, config_hash, metric_loss, out_of_range, run_league, validation_table,
)
from dynasty_sim.engine.interface import NumbaEngine
from dynasty_sim.engine.params import load_engine_params
from dynasty_sim.league.calsets import build_loop_league_set
from dynasty_sim.league.schedule import season_schedule

P, REF = load_params(), frozen_reference()


@pytest.fixture(scope="module")
def eng():
    return NumbaEngine()          # loads the frozen calibration overlay


def test_calibration_file_is_fresh():
    doc = yaml.safe_load(CALIBRATED.read_text())
    assert doc["meta"]["reference"] == REF.fingerprint, "reference changed: recalibrate"
    assert doc["meta"]["config_hash"] == config_hash(doc["params"], REF)
    assert set(doc["params"]) <= set(yaml.safe_load(open(CALIBRATED.parent / "engine.yaml"))["params"])


@pytest.mark.parametrize("seed", [2024, 777])       # 2024 = fitting league, 777 = held out
def test_human_league_inside_validation_table(eng, seed):
    """The calibration league is the Human-only league the simulation itself produces (drafted, aged, scouted,
    coached), 20 seasons after a warm-up."""
    ls = build_loop_league_set(seed)
    m = run_league(eng, ls)
    bad = out_of_range(m)
    assert not bad, {k: (round(m[k], 3), validation_table()[k]) for k in bad}


def test_calibration_uses_no_new_random_state(eng):
    """Same league set + same seeds -> identical metrics (the loss is a deterministic function of the params)."""
    ls = build_league_set(P, REF, n_seasons=3)
    assert run_league(eng, ls) == run_league(eng, ls)


def test_schedule_gives_every_team_120_games():
    rng = np.random.default_rng(0)
    for season in range(4):
        h, a = season_schedule(8, 120, rng, season)
        assert len(h) == 480 and (h != a).all()
        counts = np.bincount(np.concatenate([h, a]), minlength=8)
        assert (counts == 120).all()
        home = np.bincount(h, minlength=8)
        assert abs(home - 60).max() <= 6


def test_loss_is_zero_at_range_midpoints_and_one_at_edges():
    tab = validation_table()
    mid = {k: (0.5 * (lo + hi) if lo is not None else 0.0) for k, (lo, hi) in tab.items()}
    assert metric_loss(mid, tab) == pytest.approx(0.0)
    edge = {k: hi for k, (lo, hi) in tab.items()}
    assert metric_loss(edge, tab) == pytest.approx(sum(1.0 for lo, _ in tab.values() if lo is not None))
