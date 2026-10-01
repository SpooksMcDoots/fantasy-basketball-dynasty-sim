from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dynasty_sim.core.types import LOCI, N_LOCI, N_RACES, N_TRAITS, RACES, TRAITS

_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Params:
    h2: np.ndarray               # [T]
    inbreeding_dep: np.ndarray   # [T] SD per unit F, always positive magnitude
    dep_direction: np.ndarray    # [T] -1 lowers raw value, +1 raises it
    race_mu: np.ndarray          # [R, T] native units
    race_sd: np.ndarray          # [R, T] native units
    D: np.ndarray                # [R, R] genetic distance, zero diagonal
    base_conception: np.ndarray  # [R]
    fertile_female: np.ndarray   # [R, 2]
    locus_freq: np.ndarray       # [R, L]
    adult_age: np.ndarray        # [R] marriage / growth-complete age
    lifespan_mu: np.ndarray      # [R]
    lifespan_sd: np.ndarray      # [R]
    birth_interval: np.ndarray   # [R] minimum years between births
    K: np.ndarray                # [R] carrying capacity
    demo: dict                   # scalar demography settings (see defaults.yaml)
    timescale: np.ndarray        # [R] age_eff = 20 + (age - adult_age) / timescale
    decline_mult: np.ndarray     # [R] scales athletic decline
    learn_window: np.ndarray     # [R,3] critical learning window (lo, hi, multiplier)
    wisdom: np.ndarray           # [R] yearly rise (SD) of the vision and temperament ceiling after adulthood
    league: dict
    dev: dict
    injury: dict
    n_eff: float
    heterosis_coef: float
    outbreeding_dep_coef: float
    hybrid_fert_k: float
    f1_instability_coef: float
    f1_apply: np.ndarray         # [T] 1 where the F1 instability override applies, 0 for exempt traits
    fertility_penalty_per_F: float
    child_mortality_per_F: float
    mean_reversion: float
    tradeoff_agility: float
    tradeoff_z0: float
    viability_coef: float
    viability_z0: float
    config_hash: str

    def seed_fill_count(self, race: int) -> int:
        return int(self.demo["seed_fill"] * self.K[race])


def load_params(overrides: dict | None = None) -> Params:
    raw_d = (_DIR / "defaults.yaml").read_bytes()
    raw_r = (_DIR / "races.yaml").read_bytes()
    d, r = yaml.safe_load(raw_d), yaml.safe_load(raw_r)
    assert tuple(d["traits"]) == TRAITS and tuple(r["races"]) == RACES

    mu = np.array([[r["traits"][t][race][0] for t in TRAITS] for race in RACES], float)
    sd = np.array([[r["traits"][t][race][1] for t in TRAITS] for race in RACES], float)
    D = np.zeros((N_RACES, N_RACES))
    for key, val in r["distance"].items():
        a, b = key.split("-")
        D[RACES.index(a), RACES.index(b)] = D[RACES.index(b), RACES.index(a)] = val

    bal_path = _DIR / "races_balance.yaml"
    raw_b = bal_path.read_bytes() if bal_path.exists() else b""
    for race, shifts in (yaml.safe_load(raw_b) or {}).get("mu_shift", {}).items():
        for trait, v in shifts.items():
            mu[RACES.index(race), TRAITS.index(trait)] += float(v)
    for race, kv in (yaml.safe_load(raw_b) or {}).get("life_history", {}).items():
        r["life_history"][race].update(kv)
    g = dict(d["genetics"])
    demo = dict(d["demography"])
    if overrides:
        g.update(overrides)
    return Params(
        h2=np.array([d["traits"][t]["h2"] for t in TRAITS], float),
        inbreeding_dep=np.array([d["traits"][t]["inbreeding_dep"] for t in TRAITS], float),
        dep_direction=np.array([d["traits"][t]["dep_direction"] for t in TRAITS], float),
        race_mu=mu, race_sd=sd, D=D,
        base_conception=np.array([r["life_history"][x]["base_conception"] for x in RACES], float),
        fertile_female=np.array([r["life_history"][x]["fertile_female"] for x in RACES], float),
        locus_freq=np.array([[r["loci"][l][x] for l in LOCI] for x in RACES], float),
        adult_age=np.array([r["life_history"][x]["adult_age"] for x in RACES], float),
        lifespan_mu=np.array([r["life_history"][x]["lifespan"][0] for x in RACES], float),
        lifespan_sd=np.array([r["life_history"][x]["lifespan"][1] for x in RACES], float),
        birth_interval=np.array([r["life_history"][x]["birth_interval"] for x in RACES], float),
        K=np.array([demo["K"][x] for x in RACES], float),
        demo=demo,
        timescale=np.array([r["life_history"][x]["timescale"] for x in RACES], float),
        decline_mult=np.array([r["life_history"][x]["decline_mult"] for x in RACES], float),
        learn_window=np.array([r["life_history"][x]["learn_window"] for x in RACES], float),
        wisdom=np.array([r["life_history"][x].get("wisdom", 0.0) for x in RACES], float),
        league=dict(d["league"]), dev=dict(d["development"]), injury=dict(d["injury"]),
        n_eff=g["n_eff"], heterosis_coef=g["heterosis_coef"],
        outbreeding_dep_coef=g["outbreeding_dep_coef"], hybrid_fert_k=g["hybrid_fert_k"],
        f1_instability_coef=g["f1_instability_coef"],
        f1_apply=np.array([0.0 if t in g.get("f1_instability_exempt", []) else 1.0 for t in TRAITS]),
        fertility_penalty_per_F=g["fertility_penalty_per_F"],
        child_mortality_per_F=g["child_mortality_per_F"],
        mean_reversion=g["mean_reversion"], tradeoff_agility=g["tradeoff_agility_per_height_sd"],
        tradeoff_z0=g["tradeoff_height_z0"], viability_coef=g["viability_coef"], viability_z0=g["viability_z0"],
        config_hash=hashlib.sha256(raw_d + raw_r + raw_b + repr(sorted(g.items())).encode() + repr(sorted(demo.items(), key=str)).encode() + repr([sorted(d[k].items(), key=str) for k in ("league", "development", "injury")]).encode()).hexdigest()[:16],
    )
