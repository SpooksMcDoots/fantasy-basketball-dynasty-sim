from __future__ import annotations

from typing import Callable

# Higher seed hosts games 1, 2 and 5 of a best-of-5 (2-2-1); longer series repeat the 2-2 pattern.
def home_pattern(best_of: int) -> list[bool]:
    base = [True, True, False, False]
    return (base * best_of)[: best_of - 1] + [True]


def play_series(hi: int, lo: int, best_of: int, play_game: Callable[[int, int], tuple[int, int]]) -> int:
    """Return the winning team. play_game(home, away) -> (home_score, away_score)."""
    need = best_of // 2 + 1
    wins = {hi: 0, lo: 0}
    for hi_home in home_pattern(best_of):
        h, a = (hi, lo) if hi_home else (lo, hi)
        sh, sa = play_game(h, a)
        wins[h if sh > sa else a] += 1
        if max(wins.values()) == need:
            break
    return hi if wins[hi] >= wins[lo] else lo


def run_playoffs(seeds: list[int], best_of: int, play_game) -> dict:
    """Single-bracket playoff for any power-of-two field; `seeds` are team ids best-first.

    Each round pairs the best remaining seed with the worst (1v8, 4v5, 2v7, 3v6 for eight teams); the better
    seed hosts. Returns the champion, the finalist and the winners of the round before the final.
    """
    rank = {t: i for i, t in enumerate(seeds)}
    alive = list(seeds)
    rounds, final, series = [], None, []
    while len(alive) > 1:
        alive.sort(key=rank.get)
        n = len(alive)
        pairs = [(alive[i], alive[n - 1 - i]) for i in range(n // 2)]
        alive = [play_series(hi, lo, best_of, play_game) for hi, lo in pairs]
        series += [(hi, lo, w, len(rounds)) for (hi, lo), w in zip(pairs, alive)]   # (higher seed, lower seed, winner, round)
        rounds.append(alive)
        final = pairs[0]
    champ = alive[0]
    return {"champion": champ, "finalist": final[1] if final[0] == champ else final[0],
            "semis": tuple(rounds[-2]) if len(rounds) > 1 else (), "series": series}
