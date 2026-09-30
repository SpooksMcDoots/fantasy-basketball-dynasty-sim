"""Physical/skill traits (native units) -> per-player engine rows. Uses the frozen reference only."""
from __future__ import annotations

import numpy as np

from dynasty_sim.core.reference import Reference, derived_heights
from dynasty_sim.core.types import TRAIT_IDX
from dynasty_sim.engine import schema as S


def build_players(units: np.ndarray, ref: Reference, avail: np.ndarray | None = None) -> np.ndarray:
    """units [N, 12] in TRAITS order -> rows [N, NF]."""
    units = np.atleast_2d(units)
    t = lambda name: units[:, TRAIT_IDX[name]]
    z = lambda name: ref.z(name, t(name))
    reach, mr, _, wing = derived_heights(units)
    z_str, z_vert, z_touch = z("strength"), z("vertical"), z("touch")
    z_vis, z_temp, z_end, z_hand = z("vision"), z("temperament"), z("endurance"), z("hand")
    quick = -z("agility")
    z_mr = (mr - ref.mr_mu) / ref.mr_sd
    z_wing = (wing - ref.wing_mu) / ref.wing_sd
    mass = 95.0 * (t("height") / 198.0) ** 3 * (1.0 + 0.12 * z_str)
    handle = 0.45 * z_vis + 0.35 * quick + 0.20 * z_hand
    reb = 0.45 * z_mr + 0.35 * z_str + 0.20 * z_vert
    comp = (0.20 * z_mr + 0.15 * z_vert + 0.15 * quick + 0.10 * z_str + 0.15 * z_touch
            + 0.10 * z_vis + 0.05 * z_end + 0.05 * handle + 0.05 * reb)
    rows = np.empty((len(units), S.NF))
    rows[:, S.C_HEIGHT], rows[:, S.C_REACH], rows[:, S.C_VERT] = t("height"), reach, t("vertical")
    rows[:, S.C_QUICK], rows[:, S.C_STR], rows[:, S.C_TOUCH] = quick, z_str, z_touch
    rows[:, S.C_VISION], rows[:, S.C_TEMPER], rows[:, S.C_END] = z_vis, z_temp, z_end
    rows[:, S.C_MASS], rows[:, S.C_HAND], rows[:, S.C_WING] = mass, z_hand, z_wing
    rows[:, S.C_VERTZ], rows[:, S.C_OVR] = z_vert, 50.0 + 10.0 * comp
    rows[:, S.C_AVAIL] = 1.0 if avail is None else avail
    return rows
