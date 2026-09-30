"""Stability alarms from spec 4.4: trait drift, top-team margin, single-house title share."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynasty_sim.core.types import TRAITS
from dynasty_sim.history.dashboard import YearRecord

BURN_IN = 10                  # the founding league is a transient; drift is measured after it
MEAN_DRIFT = 0.5              # reference SD per 50 years
P99_DRIFT = 0.75
TOP_MOV = 15.0                # decade-mean best-team margin of victory
TITLE_SHARE = 0.40            # per 30 seasons
TITLE_WINDOW = 30


@dataclass(frozen=True)
class Alarm:
    kind: str
    detail: str


def _window_mean(records, field: str, lo: int, hi: int) -> np.ndarray:
    return np.mean([getattr(r, field) for r in records if lo <= r.year <= hi], axis=0)


def trait_drift(records: list[YearRecord], span: int = 50, width: int = 20) -> list[Alarm]:
    """Compare `width`-year windows `span` years apart (both after burn-in).

    Watched: the pros' mean (own-race z), the population mean, and the population 99th percentile. The pros'
    own 99th percentile stays on the dashboard but is not alarmed: with ~100 pros it is essentially the
    second-best player, so its sampling error (~0.35 SD) would swamp the 0.75 SD tolerance.
    """
    out = []
    last = records[-1].year
    for start in range(BURN_IN + 1, last - span - width + 2, width):
        a, b = (start, start + width - 1), (start + span, start + span + width - 1)
        if b[1] > last:
            break
        for field, tol in (("league_mean_z", MEAN_DRIFT), ("pop_mean_z", MEAN_DRIFT), ("pop_p99_z", P99_DRIFT)):
            d = _window_mean(records, field, *b) - _window_mean(records, field, *a)
            for i in np.flatnonzero(np.abs(d) > tol):
                out.append(Alarm("trait_drift", f"{field}[{TRAITS[i]}] moved {d[i]:+.2f} SD between years "
                                                f"{a[0]}-{a[1]} and {b[0]}-{b[1]}"))
    return out


def top_margin(records: list[YearRecord], limit: float = TOP_MOV) -> list[Alarm]:
    out = []
    late = [r for r in records if r.year > BURN_IN]
    for i in range(0, len(late), 10):
        chunk = late[i:i + 10]
        if len(chunk) == 10 and np.mean([r.top_mov for r in chunk]) > limit:
            out.append(Alarm("top_mov", f"decade {chunk[0].year}-{chunk[-1].year} mean best-team margin "
                                        f"{np.mean([r.top_mov for r in chunk]):.1f} > {limit}"))
    return out


def house_titles(records: list[YearRecord], limit: float = TITLE_SHARE, window: int = TITLE_WINDOW) -> list[Alarm]:
    out = []
    late = [r for r in records if r.year > BURN_IN]
    for i in range(0, len(late) - window + 1, 5):
        chunk = late[i:i + window]
        counts = np.bincount([r.champion_house for r in chunk])
        if counts.max() / window > limit:
            out.append(Alarm("house_titles", f"house {int(counts.argmax())} won {counts.max()}/{window} titles "
                                             f"in {chunk[0].year}-{chunk[-1].year}"))
    return out


def check_alarms(records: list[YearRecord]) -> list[Alarm]:
    return trait_drift(records) + top_margin(records) + house_titles(records)
