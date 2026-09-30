"""Who collects a rebound: a very tall player takes a large share, never nearly all of it."""
import numpy as np
from numba import njit

from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.engine import kernel
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.params import load_engine_params

PRM = load_engine_params(frozen_reference())


@njit
def _seed(s):
    np.random.seed(s)


def _shares(ratings, n=20_000):
    """Share of offensive rebounds each of the five offensive players collects, when the offence always wins the board."""
    _seed(1)
    rb = np.zeros((2, 5))
    rb[0] = ratings
    on = np.tile(np.arange(5), (2, 1))
    box = np.zeros((2, 5, S.NSTAT))
    wk = np.zeros(5)
    for _ in range(n):
        kernel._rebound(0, 50.0, 0.0, PRM, rb, on, box, wk)        # base logit 50: the offence always gets it
    got = box[0, :, S.ST_ORB]
    return got / got.sum()


def test_one_giant_among_four_small_players_gets_a_large_but_not_total_share():
    # the year-84 lineup from the 100-year run: one 250 cm Goliath (+3.2) and four smaller Elves (-1.3 to -0.2)
    share = _shares([3.24, -0.77, -0.85, -1.33, -0.17])
    assert 0.30 <= share[0] <= 0.55, share
    assert share[1:].min() >= 0.05, share                               # every perimeter player still collects some


def test_four_giants_share_the_boards_and_the_lone_small_player_still_gets_some():
    share = _shares([3.24, 3.0, 3.4, 3.1, -0.85])
    assert share[:4].max() < 0.30 and share[4] >= 0.05, share


def test_more_size_helps_but_saturates():
    a, b, c = (_shares([r, 0, 0, 0, 0])[0] for r in (0.5, 2.5, 8.0))
    assert a < b < c
    assert c - b < 0.05                                                 # past the cap, extra size adds almost nothing
    assert abs(_shares([0, 0, 0, 0, 0])[0] - 0.2) < 0.02               # equal players split evenly
