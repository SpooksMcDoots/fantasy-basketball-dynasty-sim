"""Local launcher for the history viewer: serves the page, lists saved runs, and can simulate a new run on request.

Standard library only; binds to 127.0.0.1. The page also works without this (open viewer.html and load a saved
run), but only the launcher can start a new simulation.
"""
from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dynasty_sim.output import viewer

MAX_YEARS = 300
TEAM_CHOICES = (4, 8)                           # one team per house and the population founds 8 houses; the bracket needs a power of two


class Jobs:
    """One simulation at a time; progress is the number of seasons played."""

    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self.lock = threading.Lock()
        self.state: dict = {"state": "idle"}

    def start(self, seed: int, years: int, teams: int) -> bool:
        with self.lock:
            if self.state.get("state") == "running":
                return False
            self.state = {"state": "running", "done": 0, "total": years, "seed": seed}
        threading.Thread(target=self._run, args=(seed, years, teams), daemon=True).start()
        return True

    def _run(self, seed: int, years: int, teams: int) -> None:
        try:
            from dynasty_sim.config import load_params
            from dynasty_sim.league.world import World
            P = load_params()
            P.league["n_teams"] = teams
            w = World(seed, P, history=True)
            for i in range(years):
                w.step()
                self.state["done"] = i + 1
            out = self.runs / f"viewer_s{seed}_y{years}_t{teams}"
            viewer.write_viewer(w, out)
            self.state = {"state": "done", "done": years, "total": years, "path": _rel(out / "viewer.json", self.runs)}
        except Exception as exc:                                    # surface it in the page instead of dying silently
            traceback.print_exc()
            self.state = {"state": "error", "error": f"{type(exc).__name__}: {exc}"}


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def saved_runs(runs: Path) -> list[dict]:
    out = []
    for f in sorted(runs.glob("*/viewer.json"), key=lambda p: -p.stat().st_mtime):
        try:
            with f.open(encoding="utf-8") as fh:
                head = fh.read(600)
            meta = json.loads(head[head.index('"meta":') + 7: head.index("}", head.index('"meta":')) + 1])
        except (OSError, ValueError):
            continue
        out.append({"path": _rel(f, runs), "label": f.parent.name, "seed": meta.get("seed"), "years": meta.get("years"),
                    "teams": meta.get("teams"), "mtime": int(f.stat().st_mtime)})
    return out


def make_handler(runs: Path, jobs: Jobs, current: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:                       # keep the console quiet
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", code)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(viewer.render_html().encode("utf-8"), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json({"runs": saved_runs(runs), "current": current})
            elif url.path == "/api/job":
                self._json(jobs.state)
            elif url.path == "/api/run":
                rel = parse_qs(url.query).get("path", [""])[0]
                f = (runs / rel).resolve()
                if f.name != "viewer.json" or runs.resolve() not in f.parents or not f.is_file():
                    self._json({"error": "no such run"}, 404)
                else:
                    self._send(f.read_bytes(), "application/json")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/new":
                return self._json({"error": "not found"}, 404)
            try:
                body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)) or b"{}")
                seed, years, teams = int(body["seed"]), int(body["years"]), int(body["teams"])
            except (ValueError, KeyError, TypeError):
                return self._json({"error": "need integer seed, years and teams"}, 400)
            if not (1 <= years <= MAX_YEARS) or teams not in TEAM_CHOICES or not (0 <= seed < 2 ** 32):
                return self._json({"error": f"years 1-{MAX_YEARS}, teams one of {TEAM_CHOICES}, seed 0-4294967295"}, 400)
            if not jobs.start(seed, years, teams):
                return self._json({"error": "a simulation is already running"}, 409)
            self._json({"started": True})

    return Handler


def serve(runs: Path = Path("runs"), port: int = 8765, load: Path | None = None, open_browser: bool = True) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    current = None
    if load is not None:
        try:
            current = _rel(load, runs)
        except ValueError:
            raise SystemExit(f"--load must be inside the runs folder ({runs}); copy it there or point --runs at its parent")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runs, Jobs(runs), current))
    url = f"http://127.0.0.1:{port}/"
    print(f"History viewer at {url}  (runs folder: {runs.resolve()}; Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
