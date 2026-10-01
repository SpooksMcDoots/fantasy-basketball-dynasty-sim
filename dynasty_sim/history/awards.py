"""Season value (box plus-minus with Four-Factor category weights) and yearly awards."""
from __future__ import annotations

import numpy as np

from dynasty_sim.engine import schema as S

# Four Factors weights (Oliver 40/25/20/15): shooting / turnovers / rebounding / free throws.
W_SHOOT, W_TOV, W_REB, W_FT = 0.40, 0.25, 0.20, 0.15
LG_PPP = 1.12            # league points per possession (used to price a possession)
LG_PPA = 1.10            # league points per field-goal attempt
QUALIFY_GAMES, QUALIFY_MINUTES = 40, 600


def raw_value(stats: np.ndarray) -> float:
    """Weighted possession-value of a stat line, in points (before subtracting a replacement player)."""
    s = lambda k: float(stats[k])
    shoot = (s(S.ST_PTS) - s(S.ST_FTM)) - LG_PPA * s(S.ST_FGA) + 0.5 * LG_PPP * s(S.ST_AST)
    tov = -LG_PPP * s(S.ST_TOV) + LG_PPP * s(S.ST_STL) + 0.5 * LG_PPP * s(S.ST_BLK)
    reb = 0.6 * LG_PPP * s(S.ST_ORB) + 0.25 * LG_PPP * s(S.ST_DRB)
    ft = s(S.ST_FTM) - 0.44 * LG_PPP * s(S.ST_FTA)
    return W_SHOOT * shoot + W_TOV * tov + W_REB * reb + W_FT * ft


def replacement_rate(lines: dict) -> float:
    """Per-minute value of a replacement player: the 10th percentile of qualified players' per-minute value."""
    rates = [raw_value(v[1:]) / (v[1 + S.ST_SEC] / 60.0) for v in lines.values() if v[1 + S.ST_SEC] / 60.0 >= QUALIFY_MINUTES / 3]
    return float(np.quantile(rates, 0.10)) if rates else 0.0


def season_values(lines: dict) -> dict[int, float]:
    """pid -> points added over replacement for the season. `lines[pid] = [games, stats...]`."""
    rep = replacement_rate(lines)
    return {pid: raw_value(v[1:]) - rep * v[1 + S.ST_SEC] / 60.0 for pid, v in lines.items()}


def qualified(v: np.ndarray) -> bool:
    return v[0] >= QUALIFY_GAMES and v[1 + S.ST_SEC] / 60.0 >= QUALIFY_MINUTES


def per_game(v: np.ndarray) -> dict:
    g = max(v[0], 1.0)
    return {"ppg": v[1 + S.ST_PTS] / g, "rpg": (v[1 + S.ST_ORB] + v[1 + S.ST_DRB]) / g, "apg": v[1 + S.ST_AST] / g,
            "spg": v[1 + S.ST_STL] / g, "bpg": v[1 + S.ST_BLK] / g}


MVP_WEIGHTS = (0.5, 0.3, 0.2)       # value added, points per game, team winning percentage: voters reward stars on winners


def _z(xs: dict) -> dict:
    v = np.array(list(xs.values()), float)
    sd = v.std()
    return {k: (x - v.mean()) / sd if sd > 0 else 0.0 for k, x in xs.items()}


def mvp_pick(qual: list, lines: dict, values: dict, team_of: dict, team_win: dict | None):
    """Highest blend of standardised value, scoring and team success. Without team records the third term is dropped."""
    wv, wp, ww = MVP_WEIGHTS
    zv, zp = _z({p: values[p] for p in qual}), _z({p: per_game(lines[p])["ppg"] for p in qual})
    zw = _z({p: team_win.get(team_of.get(p), 0.5) for p in qual}) if team_win else {p: 0.0 for p in qual}
    return max(qual, key=lambda p: (wv * zv[p] + wp * zp[p] + (ww * zw[p] if team_win else 0.0), -p))


ROLE_SLOTS = (("guard", 2), ("forward", 2), ("center", 1))      # All-League first team: a lineup, not the five best box scores


def assign_roles(heights: dict) -> dict:
    """pid -> guard / forward / center: the shortest third of the league's qualified players, the middle third, the tallest."""
    order = sorted(heights, key=lambda p: (heights[p], p))
    n = len(order)
    return {p: ("guard", "forward", "center")[min(2, 3 * i // max(n, 1))] for i, p in enumerate(order)}


def all_league_team(qual: list, values: dict, roles: dict | None) -> list:
    """Best five by value; with roles, the best two guards, two forwards and one center (gaps filled by the next best)."""
    ranked = sorted(qual, key=lambda p: (-values[p], p))
    if not roles:
        return ranked[:5]
    chosen: list = []
    for role, n in ROLE_SLOTS:
        chosen += [p for p in ranked if roles.get(p) == role][:n]
    chosen += [p for p in ranked if p not in chosen][: 5 - len(chosen)]
    return sorted(chosen, key=lambda p: (-values[p], p))


def yearly_awards(lines: dict, values: dict, team_of: dict, rookies: set, prev_values: dict, champion_team: int,
                  roles: dict | None = None, team_win: dict | None = None) -> dict:
    """Return {award: pid or [pids]} for one season."""
    qual = [p for p, v in lines.items() if qualified(v)]
    if not qual:
        return {}
    best = lambda pool, key: max(pool, key=lambda p: (key(p), -p)) if pool else None
    out = {"mvp": mvp_pick(qual, lines, values, team_of, team_win),
           "scoring": best(qual, lambda p: per_game(lines[p])["ppg"]),
           "dpoy": best(qual, lambda p: (lines[p][1 + S.ST_STL] + lines[p][1 + S.ST_BLK] + 0.25 * lines[p][1 + S.ST_DRB])
                        / max(lines[p][1 + S.ST_SEC] / 60.0, 1.0)),
           "all_league": all_league_team(qual, values, roles)}
    rk = [p for p in qual if p in rookies]
    if rk:
        out["roy"] = best(rk, lambda p: values[p])
    imp = [p for p in qual if p in prev_values]
    if imp:
        out["most_improved"] = best(imp, lambda p: values[p] - prev_values[p])
    champs = [p for p in qual if team_of.get(p) == champion_team]
    if champs:
        out["finals_mvp"] = best(champs, lambda p: values[p])
    return out
