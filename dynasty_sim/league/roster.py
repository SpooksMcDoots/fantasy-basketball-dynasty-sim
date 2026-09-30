from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from dynasty_sim.engine.interface import CoachParams


@dataclass
class Team:
    id: int
    house_id: int
    name: str
    roster: list = field(default_factory=list)       # person ids, at most roster_max
    coach: CoachParams = field(default_factory=CoachParams)
    parent_team_id: Optional[int] = None             # feeder/vassal hooks (logic comes later)
    tier: int = 0


def coach_from_head(person, scale: float = 1.0) -> CoachParams:
    """A house head's own temperament and skills set the team's coaching style (small, bounded effects)."""
    import numpy as np
    from dynasty_sim.core.types import TRAIT_IDX
    z = person.genome.A + person.genome.E_perm
    g = lambda t: float(z[TRAIT_IDX[t]])
    clip = lambda x, lim: float(np.clip(x, -lim, lim))
    return CoachParams(
        style_bias=(scale * clip(0.02 * g("height"), 0.04), scale * clip(-0.01 * g("touch"), 0.03),
                    scale * clip(0.02 * g("touch"), 0.04)),
        pace_pref=scale * clip(0.25 * g("temperament"), 0.5),
        rotation_depth=10 + int(round(scale * clip(g("vision"), 1.0))),
        foul_aggression=scale * clip(0.2 * g("temperament"), 0.4),
    )
