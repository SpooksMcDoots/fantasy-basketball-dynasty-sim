"""Engine parameter vector. Names -> integer constants P_<NAME> so the numba kernel indexes a flat array."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

_YAML = Path(__file__).parent.parent / "config" / "engine.yaml"

YAML_NAMES = (
    "pace_mean", "pace_shape", "pace_quick_w", "pace_pref_w", "second_len", "reach_sat", "z_sat", "pace_game_sd", "shoot_game_sd", "foul_game_sd", "score_flow", "ev_mid", "ev_three",
    "nsf_base", "nsf_aggr", "bonus_at",
    "tov", "tov_slope", "tov_temper", "steal_share", "steal_slope",
    "usage_touch", "usage_vision", "usage_temper", "usage_sat", "tau",
    "make_rim", "make_mid", "make_three", "ft",
    "beta_touch_rim", "beta_touch_mid", "beta_touch_three", "beta_touch_ft",
    "beta_c_rim", "beta_c_mid", "beta_c_three", "size_rim", "size_clip",
    "c_reach_rim", "c_reach_mid", "c_reach_three", "c_quick_rim", "c_quick_mid", "c_quick_three",
    "c_off_three",
    "blk_rim", "blk_mid", "blk_three", "blk_slope",
    "sf_rim", "sf_mid", "sf_three", "sf_str", "sf_quick",
    "orb", "orb_ft", "orb_slope", "orb_three", "reb_pick", "reb_sat",
    "ast_rim", "ast_mid", "ast_three", "ast_vision", "ast_pick",
    "home_make", "home_foul", "clutch",
    "energy_drain", "drain_end", "bench_rec", "fat_factor", "fat_floor", "sub_energy", "sub_margin",
)
# probabilities whose logit the kernel needs
LOGIT_NAMES = ("tov", "make_rim", "make_mid", "make_three", "ft", "blk_rim", "blk_mid", "blk_three",
               "sf_rim", "sf_mid", "sf_three", "orb", "orb_ft", "ast_rim", "ast_mid", "ast_three")
# constants filled in from the frozen reference
REF_NAMES = ("mr_mu", "mr_sd", "rel_mu", "reach_off")

ALL_NAMES = YAML_NAMES + tuple("lg_" + n for n in LOGIT_NAMES) + REF_NAMES
for _i, _n in enumerate(ALL_NAMES):
    globals()["P_" + _n.upper()] = _i
NPAR = len(ALL_NAMES)


_CALIBRATED = _YAML.parent / "engine_calibrated.yaml"


def load_engine_params(ref, overrides: dict | None = None, calibrated: bool = True) -> np.ndarray:
    """Defaults from engine.yaml, then the frozen calibration overlay (if present), then overrides."""
    raw = yaml.safe_load(_YAML.read_text())["params"]
    if calibrated and _CALIBRATED.exists():
        raw = {**raw, **yaml.safe_load(_CALIBRATED.read_text())["params"]}
    if overrides:
        raw = {**raw, **overrides}
    missing, extra = set(YAML_NAMES) - set(raw), set(raw) - set(YAML_NAMES)
    if missing or extra:
        raise ValueError(f"engine.yaml mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    vals = {n: float(raw[n]) for n in YAML_NAMES}
    for n in LOGIT_NAMES:
        p = min(max(vals[n], 1e-6), 1 - 1e-6)
        vals["lg_" + n] = float(np.log(p / (1 - p)))
    vals.update(mr_mu=ref.mr_mu, mr_sd=ref.mr_sd, rel_mu=ref.rel_mu, reach_off=ref.reach_off)
    return np.array([vals[n] for n in ALL_NAMES], dtype=np.float64)
