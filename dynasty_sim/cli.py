"""Command line: run a full world (population + league + dashboards) and write deterministic outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynasty_sim.config import load_params
from dynasty_sim.core.reference import frozen_reference, reference_to_json
from dynasty_sim.core.types import TRAITS
from dynasty_sim.engine.params import _CALIBRATED
from dynasty_sim.genetics.genome import adult_phenotype_z
from dynasty_sim.history import alarms as A
from dynasty_sim.history.dashboard import decade_table
from dynasty_sim.league.world import World


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in o.ravel()] if o.ndim == 1 else [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(years: int, seed: int, out: Path, teams: int = 8, history: bool = True, parquet: bool = False) -> World:
    P = load_params()
    P.league["n_teams"] = teams
    out.mkdir(parents=True, exist_ok=True)
    w = World(seed, P, history=history).run(years)
    ref = frozen_reference()
    alarms = A.check_alarms(w.records) if years > A.BURN_IN else []
    manifest = {**w.pop.tree.manifest(), "config_hash": P.config_hash, "engine_calibration": _CALIBRATED.read_text().split("config_hash: ")[1][:16],
                "reference": ref.fingerprint, "years": years, "teams": teams, "n_people": len(w.pop.people),
                "champions": [r.champion_house for r in w.records], "alarms": [f"{a.kind}: {a.detail}" for a in alarms]}
    people = [{"id": p.id, "sex": p.sex, "birth_year": p.birth_year, "mother": p.mother_id, "father": p.father_id,
               "house": p.house_id, "F": round(p.genome.F, 6), "ancestry": [round(float(a), 6) for a in p.genome.ancestry],
               "z": [round(float(v), 6) for v in adult_phenotype_z(p.genome, P)]} for p in w.pop.people.values()]
    if history:
        from dynasty_sim.output.export import export_history
        summary = export_history(w, out / "history", parquet=parquet)
        from dynasty_sim.output.viewer import write_viewer
        write_viewer(w, out)
        led = w.ledger
        manifest["history"] = {"pages": summary["pages"], "unresolved": summary["problems"], "arcs": summary["arcs"],
                               "events": len(led.log.events), "hall_of_fame": len(led.hof.players),
                               "rivalries": len(led.rivalry.flagged), "record_breaks": len(led.log.by_kind("record"))}
    (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))
    (out / "people.json").write_text(json.dumps(people, sort_keys=True))
    (out / "reference.json").write_text(json.dumps(reference_to_json(ref), sort_keys=True, indent=1))
    (out / "dashboard.json").write_text(json.dumps(_jsonable({"traits": TRAITS, "decades": decade_table(w.records)}),
                                                   sort_keys=True))
    return w


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="dynasty_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--years", type=int, default=1)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--teams", type=int, default=8)
    r.add_argument("--out", type=Path, default=Path("runs"))
    r.add_argument("--no-possession-log", action="store_true", help="accepted for spec compatibility; logs are off by default")
    r.add_argument("--no-history", action="store_true", help="skip the Markdown history pages")
    r.add_argument("--parquet", action="store_true", help="also write player_seasons/events/team_seasons Parquet tables")
    sw = sub.add_parser("sweep", help="run many seeds in parallel and report dashboards/alarms")
    sw.add_argument("--seeds", type=int, nargs=2, default=[0, 20], metavar=("FIRST", "LAST_EXCLUSIVE"))
    sw.add_argument("--years", type=int, default=100)
    sw.add_argument("--workers", type=int, default=4)
    v = sub.add_parser("view", help="open the history viewer in a browser (load a saved run or start a new one)")
    v.add_argument("--runs", type=Path, default=Path("runs"), help="folder holding saved runs (each with a viewer.json)")
    v.add_argument("--load", type=Path, help="a viewer.json inside the runs folder to open straight away")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "view":
        from dynasty_sim.output.viewer_server import serve
        serve(args.runs, args.port, args.load, open_browser=not args.no_browser)
        return
    if args.cmd == "sweep":
        from dynasty_sim.history.sweep import report, sweep
        res = sweep(range(*args.seeds), args.years, workers=args.workers)
        print(report(res))
        for r in res:
            for a in r.alarms:
                print(f"  seed {r.seed}: {a}")
        return
    w = run(args.years, args.seed, args.out, args.teams, history=not args.no_history, parquet=args.parquet)
    print(f"ran {args.years} years, seed {args.seed}: {len(w.pop.people)} people, "
          f"{len(w.league.history)} seasons, champions {[r.champion_house for r in w.records][-5:]}")


if __name__ == "__main__":
    main()
