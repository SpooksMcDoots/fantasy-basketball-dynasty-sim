"""Scouting value: what a front office believes a player is worth, fitted to actual simulated winning contribution.

The engine's game composite (`C_OVR`) only drives coaching rotations. Drafting and roster decisions use this
regression instead, so a player who contributes more than his composite suggests (a rim protector, a rebounder)
is recognised, and no team gains an edge from a market blind spot. Weights are fitted once and frozen in
config/scouting.yaml with the reference fingerprint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from dynasty_sim.config import Params
from dynasty_sim.core.reference import Reference
from dynasty_sim.engine import schema as S

_PATH = Path(__file__).parent.parent / "config" / "scouting.yaml"
FEATURES = ("z_reach", "z_vert", "quick", "str", "touch", "vision", "temper", "end", "hand", "wing", "z_mass", "z_reach_sq")
_SAT = 2.5          # SD at which a physical edge stops adding value (mirrors the engine's diminishing returns)


def features(rows: np.ndarray, ref: Reference) -> np.ndarray:
    """rows [N, NF] -> [N, len(FEATURES)] with saturating transforms."""
    zr = (rows[:, S.C_REACH] + rows[:, S.C_VERT] - ref.mr_mu) / ref.mr_sd
    sat = lambda x: _SAT * np.tanh(x / _SAT)
    mass_z = (rows[:, S.C_MASS] - 95.0) / 25.0
    return np.column_stack([
        sat(zr), sat(rows[:, S.C_VERTZ]), sat(rows[:, S.C_QUICK]), sat(rows[:, S.C_STR]), rows[:, S.C_TOUCH],
        rows[:, S.C_VISION], rows[:, S.C_TEMPER], rows[:, S.C_END], rows[:, S.C_HAND], sat(rows[:, S.C_WING]),
        sat(mass_z), sat(zr) ** 2 / _SAT,
    ])


def load_weights() -> np.ndarray:
    doc = yaml.safe_load(_PATH.read_text())
    return np.array([doc["weights"][f] for f in FEATURES], float)


def scout_value(rows: np.ndarray, ref: Reference, w: np.ndarray | None = None) -> np.ndarray:
    """Points of expected margin per game (relative to the reference league) scaled to a rating-like number."""
    w = load_weights() if w is None else w
    return 50.0 + features(rows, ref) @ w


def fit_weights(eng, P: Params, ref: Reference, n_games: int = 24_000, seed: int = 99, ridge: float = 30.0):
    """Regress simulated game margin on the difference of two teams' mean rotation features."""
    from dynasty_sim.engine.composites import build_players
    from dynasty_sim.league.pool import selected_units
    rng = np.random.default_rng(seed)
    n_teams = 4000
    # a spread of populations so the fit sees the full range of physical profiles
    race_mix = rng.choice(["human", "goliath", "elf"], size=n_teams * 13, p=[0.5, 0.25, 0.25])
    rows = np.empty((n_teams * 13, S.NF))
    for r in ("human", "goliath", "elf"):
        idx = np.flatnonzero(race_mix == r)
        rows[idx] = build_players(selected_units(len(idx), rng, P, ref, r), ref)
    teams = rows.reshape(n_teams, 13, S.NF)
    a, b = rng.integers(0, n_teams, n_games), rng.integers(0, n_teams, n_games)
    keep = a != b
    a, b = a[keep], b[keep]
    PL = np.zeros((len(a), 2, 13, S.NF)); PL[:, 0], PL[:, 1] = teams[a], teams[b]
    CO = np.zeros((len(a), 2, S.NCO)); CO[:, :, S.CO_DEPTH] = 10
    _, SC, _ = eng.simulate_arrays(PL, np.full((len(a), 2), 13), CO, np.arange(len(a)) + 5_000_000)
    margin = (SC[:, 0] - SC[:, 1]).astype(float)

    def team_feat(t):        # mean of the top-8 by composite = the rotation
        out = np.empty((len(t), len(FEATURES)))
        for i, tm in enumerate(t):
            top = tm[np.argsort(-tm[:, S.C_OVR])[:8]]
            out[i] = features(top, ref).mean(0)
        return out
    F = team_feat(teams[a]) - team_feat(teams[b])
    X = np.vstack([F, -F]); y = np.concatenate([margin, -margin])          # symmetric: no intercept, no home bias
    w = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)
    pred = F @ w
    r2 = 1 - np.sum((margin - pred) ** 2) / np.sum((margin - margin.mean()) ** 2)
    scale = 10.0 / max(float(np.std(features(rows, ref) @ w)), 1e-9)      # 1 SD of value across the pool = 10 points
    return w * scale, float(r2)


def write(w: np.ndarray, r2: float, ref: Reference) -> None:
    doc = {"meta": {"reference": ref.fingerprint, "r2": round(r2, 4)},
           "weights": {f: round(float(v), 6) for f, v in zip(FEATURES, w)}}
    _PATH.write_text("# Frozen output of dynasty_sim.league.scouting.fit_weights. Do not hand-edit.\n" + yaml.safe_dump(doc, sort_keys=False))
