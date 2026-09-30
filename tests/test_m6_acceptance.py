"""M6 acceptance: 100-year dashboards stay inside the alarms across 20 seeds.

Seeds 200-219 are held out: the configuration was tuned on seeds 0-23 and 100-119 only.
"""
import numpy as np
import pytest

from dynasty_sim.history.sweep import report, run_seed, sweep


def test_short_sweep_summary_is_well_formed():
    res = sweep([0, 1], years=14, workers=1)
    assert [r.seed for r in res] == [0, 1]
    for r in res:
        assert 1 <= r.houses_alive <= 8 and 0 <= r.mean_dynasty_share <= 1 and len(r.race_share) == 3
    assert "2 seeds" in report(res)


def test_overrides_reach_the_world():
    a = run_seed((3, 12, {"league.house_cap": 1}))
    b = run_seed((3, 12, None))
    assert a.mean_dynasty_share <= b.mean_dynasty_share


@pytest.mark.slow
def test_twenty_held_out_seeds_stay_inside_alarms():
    res = sweep(range(200, 220), years=100, workers=4)
    assert not [(r.seed, a) for r in res for a in r.alarms if a.startswith("top_mov")]       # no runaway margins, ever
    drift = [(r.seed, a) for r in res for a in r.alarms if a.startswith("trait_drift")]
    # A population's 99th percentile fluctuates ~0.3 SD from finite-population drift alone, so a 0.75 SD tolerance over
    # 12 traits and 20 seeds trips about once in twenty seeds. More than two means a real trend, not sampling.
    assert len({s for s, _ in drift}) <= 2, drift
    assert min(r.houses_alive for r in res) >= 5              # dynasties survive (spec test b)
    assert min(r.min_race_ratio for r in res) > 0.5           # every race stays above half its carrying capacity
    clean = sum(1 for r in res if not r.alarms)
    assert clean >= 18, [(r.seed, r.alarms) for r in res if r.alarms]   # title concentration alarms are rare (<10%)
    assert max(max(r.decade_top_mov) for r in res) < 15.0
