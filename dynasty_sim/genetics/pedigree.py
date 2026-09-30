"""Tabular coancestry method. Requires parent ids < child ids."""
from __future__ import annotations

import sys
from typing import Optional

sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))


class Pedigree:
    def __init__(self) -> None:
        self._parents: dict[int, tuple[Optional[int], Optional[int]]] = {}
        self._F: dict[int, float] = {}
        self._memo: dict[tuple[int, int], float] = {}

    def add(self, pid: int, mother: Optional[int] = None, father: Optional[int] = None) -> float:
        """Register a person and return their inbreeding coefficient F = f(mother, father)."""
        for p in (mother, father):
            if p is not None and (p not in self._parents or p >= pid):
                raise ValueError(f"parent {p} must be registered with id < {pid}")
        F = self.coancestry(mother, father) if mother is not None and father is not None else 0.0
        self._parents[pid] = (mother, father)
        self._F[pid] = F
        return F

    def F(self, pid: int) -> float:
        return self._F[pid]

    def coancestry(self, x: int, y: int) -> float:
        if x == y:
            return 0.5 * (1.0 + self._F[x])
        if x > y:
            x, y = y, x  # y is the younger, so it cannot be an ancestor of x
        key = (x, y)
        v = self._memo.get(key)
        if v is None:
            sire, dam = self._parents[y]
            v = 0.5 * ((self.coancestry(x, sire) if sire is not None else 0.0)
                       + (self.coancestry(x, dam) if dam is not None else 0.0))
            self._memo[key] = v
        return v
