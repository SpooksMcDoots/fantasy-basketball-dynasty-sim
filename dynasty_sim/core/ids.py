class IdAllocator:
    """Monotonic ids. Parents always get smaller ids than children (pedigree relies on it)."""

    def __init__(self, start: int = 1):
        self._next = start

    def next(self) -> int:
        i = self._next
        self._next += 1
        return i
