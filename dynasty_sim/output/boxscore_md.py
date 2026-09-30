"""Box scores as Markdown tables."""
from __future__ import annotations

from dynasty_sim.engine import schema as S
from dynasty_sim.history.ledger import GameSnapshot

HEAD = "| Player | MIN | PTS | REB | AST | STL | BLK | TOV | PF | FG | 3P | FT |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def render_box(r, snap: GameSnapshot, title: str, prefix: str = "../") -> str:
    lg = r.world.league
    (ta, tb), (sa, sb) = snap.teams, snap.score
    out = [f"# {title}", "",
           f"**{r.t(ta)}** {sa} — {sb} **{r.t(tb)}** ({'playoffs' if snap.playoff else 'regular season'}, year {snap.year})", ""]
    for side, team in enumerate((ta, tb)):
        out += [f"## {r.t(team)}", "", HEAD]
        tot = [0.0] * S.NSTAT
        for i, pid in enumerate(snap.rosters[side]):
            row = snap.box[side, i]
            if row[S.ST_SEC] <= 0:
                continue
            tot = [x + y for x, y in zip(tot, row)]
            out.append(_line(r.p(pid, prefix), row))
        out += [_line("**Team**", tot), ""]
    return "\n".join(out)


def _line(label: str, row) -> str:
    g = lambda k: int(round(row[k]))
    return (f"| {label} | {row[S.ST_SEC] / 60:.0f} | {g(S.ST_PTS)} | {g(S.ST_ORB) + g(S.ST_DRB)} | {g(S.ST_AST)} | {g(S.ST_STL)} | "
            f"{g(S.ST_BLK)} | {g(S.ST_TOV)} | {g(S.ST_PF)} | {g(S.ST_FGM)}-{g(S.ST_FGA)} | {g(S.ST_TPM)}-{g(S.ST_TPA)} | "
            f"{g(S.ST_FTM)}-{g(S.ST_FTA)} |")
