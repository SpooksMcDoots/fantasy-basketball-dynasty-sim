from __future__ import annotations

import numpy as np


def rank_teams(wins: np.ndarray, mov: np.ndarray) -> list[int]:
    """Best first: wins, then margin of victory, then team id (deterministic)."""
    return sorted(range(len(wins)), key=lambda t: (-wins[t], -mov[t], t))


def draft_order(wins: np.ndarray, mov: np.ndarray) -> list[int]:
    """Reverse standings: worst team picks first."""
    return rank_teams(wins, mov)[::-1]
