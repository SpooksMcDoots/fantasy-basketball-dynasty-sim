"""Season schedule: every pair meets equally often, plus one matching round so each team plays exactly `games`."""
from __future__ import annotations

import numpy as np


def season_schedule(n_teams: int, games: int, rng: np.random.Generator, season: int = 0):
    """Return (home[G], away[G]) team indices in play order, G = n_teams * games / 2."""
    pairs = [(a, b) for a in range(n_teams) for b in range(a + 1, n_teams)]
    per_pair, extra_per_team = divmod(games, n_teams - 1)
    home, away = [], []
    for a, b in pairs:
        for k in range(per_pair):
            h, w = (a, b) if k % 2 == 0 else (b, a)
            home.append(h)
            away.append(w)
    # extra rounds: each is a perfect matching (round-robin circle method), rotated by season
    for e in range(extra_per_team):
        rnd = (season + e) % (n_teams - 1)
        ring = [0] + [1 + (i + rnd) % (n_teams - 1) for i in range(n_teams - 1)]
        for i in range(n_teams // 2):
            a, b = ring[i], ring[n_teams - 1 - i]
            h, w = (a, b) if (season + i) % 2 == 0 else (b, a)
            home.append(h)
            away.append(w)
    order = rng.permutation(len(home))
    return np.array(home, np.int64)[order], np.array(away, np.int64)[order]
