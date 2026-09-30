import numpy as np

from dynasty_sim.cli import main
from dynasty_sim.core.rng import RngTree


def test_same_seed_byte_identical(tmp_path):
    for name in ("a", "b"):
        main(["run", "--years", "1", "--seed", "42", "--out", str(tmp_path / name)])
    for f in ("manifest.json", "people.json"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_different_seed_differs(tmp_path):
    main(["run", "--years", "1", "--seed", "1", "--out", str(tmp_path / "a")])
    main(["run", "--years", "1", "--seed", "2", "--out", str(tmp_path / "b")])
    assert (tmp_path / "a" / "people.json").read_bytes() != (tmp_path / "b" / "people.json").read_bytes()


def test_streams_independent_of_each_other():
    t = RngTree(7)
    before = t.season("engine", 3).random(5)
    t.stream("narrative").random(1000)           # narrative code consumes draws...
    t.season("genetics", 3).random(1000)
    assert np.array_equal(before, RngTree(7).season("engine", 3).random(5))  # ...engine unchanged


def test_game_seed_stable_and_distinct():
    t = RngTree(7)
    assert t.game_seed(1, 5) == RngTree(7).game_seed(1, 5)
    assert len({t.game_seed(1, g) for g in range(200)}) == 200
