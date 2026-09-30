"""Player pools for calibration and tests: draw trait vectors, keep draft-selected ones, form teams."""
from __future__ import annotations

import numpy as np

from dynasty_sim.config import Params
from dynasty_sim.core.reference import ARCHETYPES, Reference, draft_z_score, select_mask
from dynasty_sim.core.types import RACE_IDX
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players


def pool_units(n: int, rng: np.random.Generator, P: Params, race: str = "human") -> np.ndarray:
    """n trait vectors [n, 12] in native units from a single population's trait table."""
    r = RACE_IDX[race]
    return P.race_mu[r] + P.race_sd[r] * rng.standard_normal((n, P.race_mu.shape[1]))


def selected_units(n: int, rng: np.random.Generator, P: Params, ref: Reference,
                   race: str = "human") -> np.ndarray:
    """n independent draws from the draft-selected league: pick a positional archetype, then reject on its cut."""
    shares = np.array([a[1] for a in ARCHETYPES])
    counts = rng.multinomial(n, shares / shares.sum())
    out = []
    for a, k in enumerate(counts):
        got = np.empty((0, P.race_mu.shape[1]))
        while len(got) < k:
            u = pool_units(max(4000, int(3 * k / ARCHETYPES[a][2])), rng, P, race)
            got = np.vstack([got, u[select_mask(u, P, ref.draft_cuts, a, race)]])
        out.append(got[:k])
    units = np.vstack(out)
    return units[rng.permutation(len(units))]


def selected_players(n: int, rng: np.random.Generator, P: Params, ref: Reference,
                     race: str = "human") -> np.ndarray:
    """Engine rows [n, NF] for n draft-selected players, best game composite first."""
    rows = build_players(selected_units(n, rng, P, ref, race), ref)
    return rows[np.argsort(-rows[:, S.C_OVR])]


def deal_teams(rows: np.ndarray, n_teams: int, per_team: int = 13) -> list[np.ndarray]:
    """Snake-deal players (best first) into balanced teams."""
    teams: list[list[int]] = [[] for _ in range(n_teams)]
    for i in range(n_teams * per_team):
        rnd, pos = divmod(i, n_teams)
        teams[pos if rnd % 2 == 0 else n_teams - 1 - pos].append(i)
    return [rows[idx] for idx in teams]


def random_teams(rows: np.ndarray, n_teams: int, rng: np.random.Generator, per_team: int = 13) -> np.ndarray:
    """Random (unbalanced) deal: [n_teams, per_team, NF]. Strength spread comes from the draw itself."""
    idx = rng.permutation(len(rows))[: n_teams * per_team].reshape(n_teams, per_team)
    return rows[idx]


def selected_founders(n: int, race: str, rng: np.random.Generator, P: Params, ref: Reference) -> list:
    """n unrelated pure-race founder genomes drawn from the pro-caliber tail of `race` (own-race z-scores)."""
    from dynasty_sim.core.types import N_LOCI, N_RACES, N_TRAITS, Genome
    r = RACE_IDX[race]
    shares = np.array([a[1] for a in ARCHETYPES])
    counts = rng.multinomial(n, shares / shares.sum())
    out = []
    for a, k in enumerate(counts):
        kept = 0
        while kept < k:
            m = max(4000, int(3 * (k - kept) / ARCHETYPES[a][2]))
            A = rng.normal(0.0, np.sqrt(P.h2), size=(m, N_TRAITS))
            E = rng.normal(0.0, np.sqrt(1.0 - P.h2), size=(m, N_TRAITS))
            ok = np.flatnonzero(draft_z_score(A + E, a) >= ref.draft_cuts[a])[: k - kept]
            for i in ok:
                anc = np.zeros(N_RACES)
                anc[r] = 1.0
                loci = (rng.random((N_LOCI, 2)) < P.locus_freq[r][:, None]).astype(np.uint8)
                out.append(Genome(A=A[i], loci=loci, ancestry=anc, F=0.0, E_perm=E[i], hyb_gen=0))
            kept += len(ok)
    return [out[i] for i in rng.permutation(len(out))]
