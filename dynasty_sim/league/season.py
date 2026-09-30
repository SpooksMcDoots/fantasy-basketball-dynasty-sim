"""League orchestration: offseason (deaths, development, retirement, draft) then a day-by-day season with injuries."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.core.types import Event, RACES
from dynasty_sim.development import injury as inj
from dynasty_sim.development.aging import age_eff
from dynasty_sim.development.growth import develop_year, new_state
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players
from dynasty_sim.engine.interface import NumbaEngine
from dynasty_sim.league import draft as D
from dynasty_sim.league.playoffs import run_playoffs
from dynasty_sim.league.roster import Team, coach_from_head
from dynasty_sim.league.scouting import scout_value
from dynasty_sim.league.standings import draft_order, rank_teams
from dynasty_sim.population.commoners import draw_death_year
from dynasty_sim.population.demography import Population


@dataclass
class SeasonResult:
    year: int
    wins: np.ndarray
    losses: np.ndarray
    mov: np.ndarray
    champion: int = -1
    finalist: int = -1
    totals: np.ndarray = field(default_factory=lambda: np.zeros(S.NSTAT))   # league-wide box sums
    poss: int = 0
    n_games: int = 0
    seconds: float = 0.0
    series: list = field(default_factory=list)            # playoff series: (higher seed, lower seed, winner, round)
    seeds: list = field(default_factory=list)
    team_box: np.ndarray = None            # [n_teams, NSTAT] regular-season totals
    opp_box: np.ndarray = None             # [n_teams, NSTAT] what opponents did against each team
    team_poss: np.ndarray = None
    team_games: np.ndarray = None
    minutes: list = None                   # per team: {pid: seconds}
    anc: dict = field(default_factory=dict)   # pid -> ancestry vector, for players who played
    home_margin_sum: float = 0.0
    home_wins: int = 0
    injuries: int = 0
    injury_exposures: int = 0
    games_missed: list = field(default_factory=list)      # per injury that cost games


def circle_rounds(n: int) -> list[list[tuple[int, int]]]:
    """Round-robin matchings (each team plays once per round) by the circle method."""
    ring = list(range(n))
    rounds = []
    for r in range(n - 1):
        rounds.append([(ring[i], ring[n - 1 - i]) for i in range(n // 2)])
        ring = [ring[0]] + [ring[-1]] + ring[1:-1]
    return rounds


class League:
    def __init__(self, pop: Population, eng: NumbaEngine | None = None, ref=None):
        self.pop, self.P, self.tree = pop, pop.P, pop.tree
        self.ref = ref or frozen_reference()
        self.eng = eng or NumbaEngine(self.ref)
        L = self.P.league
        self.teams = [Team(i, house_id=i + 1, name=f"Team {i + 1}") for i in range(L["n_teams"])]
        self.persons: dict = {}          # pid -> Person for every current pro (house members alias pop.people)
        self.states: dict = {}           # pid -> PlayerState
        self.pool: list = []
        self.history: list[SeasonResult] = []
        self.year = 0
        self.rng = self.tree.season("league", 0)
        self._rows = [None] * L["n_teams"]
        self._game_id = 0
        self.last_wins = self.last_mov = None
        self.emergency_signings = 0
        self.capture_games: list | None = None   # when a list, every regular-season game's input arrays are appended (calibration)
        self.observers: list = []        # objects with on_game(league, pair, rosters, box, score, regular)

    # ---- helpers -------------------------------------------------------------------
    def _g_rng(self):
        return self.tree.season("genetics", 1_000_000 + self.year)

    def age(self, pid: int) -> int:
        return self.year - self.persons[pid].birth_year

    def roster_states(self, team):
        return [self.states[p] for p in team.roster]

    def _log(self, kind, house_id, actors=(), detail=""):
        self.pop.events.append(Event(self.year, kind, house_id, tuple(actors), detail))

    # ---- founding --------------------------------------------------------------------
    def found(self) -> None:
        """Year 0: house pros join their teams, prospects of mixed ages fill the rest."""
        L = self.P.league
        self.rng = self.tree.season("league", 0)
        D.house_auto_join(self)
        D.release_pool(self)
        need = L["n_teams"] * L["roster_max"] - sum(len(t.roster) for t in self.teams)
        D.generate_prospects(self, need, 0, 11, self.rng, self._g_rng(), develop_to_age=True)
        self.pool = [p for p, s in self.states.items() if s.team_id is None]
        D.run_draft(self, list(self.rng.permutation(L["n_teams"])))
        D.release_pool(self)
        self._fill_min()
        self._rebuild_rows()

    # ---- yearly cycle ------------------------------------------------------------------
    def step_year(self) -> SeasonResult:
        self.pop.step()
        self.year = self.pop.year
        self.rng = self.tree.season("league", self.year)
        self._deaths()
        self._develop()
        self._retire()
        self._offseason_roster()
        self._update_coaches()
        self._rebuild_rows()
        res = self._play_season()
        self._playoffs(res)
        self._season_end(res)
        self.history.append(res)
        return res

    def _update_coaches(self) -> None:
        for team in self.teams:
            head = self.pop.houses[team.house_id].head_id
            person = self.pop.people.get(head) if head is not None else None
            if person is not None:
                team.coach = coach_from_head(person, self.P.league["coach_effect"])

    def _remove(self, pid: int) -> None:
        """Take a player out of the league (roster, state, pro registry). The Person itself is untouched."""
        st = self.states.pop(pid, None)
        if st is not None and st.team_id is not None:
            team = self.teams[st.team_id]
            if pid in team.roster:
                team.roster.remove(pid)
        self.persons.pop(pid, None)

    def _deaths(self) -> None:
        for pid in list(self.states):
            p = self.persons[pid]
            dead = (not p.alive) if pid in self.pop.people else (p.death_year is not None and p.death_year <= self.year)
            if dead:
                self._log("player_death", p.house_id, (pid,), "died in service")
                self._remove(pid)

    def _develop(self) -> None:
        for pid, st in self.states.items():
            develop_year(st, self.age(pid) - 1, self.P, self.rng)
            st.seasons += 1
            st.injury_type, st.games_left = "", 0            # summer heals everything

    def _retire(self) -> None:
        L = self.P.league
        pids = list(self.states)
        if not pids:
            return
        from dynasty_sim.league.scouting import scout_value
        ovr = scout_value(build_players(np.array([self.states[p].units() for p in pids]), self.ref), self.ref)
        floor = np.quantile(ovr, L["replacement_pct"])
        for pid, o in zip(pids, ovr):
            st, p = self.states[pid], self.persons[pid]
            ae = age_eff(self.age(pid), st.race, self.P)
            prob = 1.0 / (1.0 + np.exp(-(ae - L["retire_center"]) / L["retire_width"]))
            if o < floor and ae >= 28:
                st.low_years += 1
                prob = max(prob, 0.35)
            if self.rng.random() < prob:
                self._log("retire", p.house_id, (pid,), f"age {self.age(pid)}")
                commoner = pid not in self.pop.people
                self._remove(pid)
                if commoner and p.alive and self.rng.random() < L["join_population_prob"]:
                    self.pop.people[pid] = p                     # retired commoner joins the mate pool

    def _offseason_roster(self) -> None:
        L = self.P.league
        self.pool = []
        D.expire_contracts(self)
        D.house_auto_join(self)
        D.population_entrants(self)
        n_open = sum(L["roster_max"] - len(t.roster) for t in self.teams)
        if n_open > 0:
            per = max(1, int(np.ceil(L["prospects_per_team"] * L["n_teams"])))
            D.generate_prospects(self, per, 0, 2, self.rng, self._g_rng())
            self.pool = [p for p, s in self.states.items() if s.team_id is None]
            order = (draft_order(self.last_wins, self.last_mov) if self.last_wins is not None
                     else list(self.rng.permutation(L["n_teams"])))
            picks = D.run_draft(self, order)
            for t, pid in picks:
                self._log("draft", self.teams[t].house_id, (pid,), f"team {t}")
        D.release_pool(self)
        self._fill_min()

    def _new_free_agent(self, team) -> None:
        pid = D.generate_prospects(self, 1, 0, 6, self.rng, self._g_rng())[0]
        D.sign(self, team, pid)

    def _fill_min(self) -> None:
        for team in self.teams:
            while len(team.roster) < self.P.league["roster_min"]:
                self._new_free_agent(team)

    # ---- game arrays -------------------------------------------------------------------------
    def _rebuild_rows(self) -> None:
        for team in self.teams:
            units = np.array([self.states[p].units() for p in team.roster])
            rows = np.zeros((S.MAX_ROSTER, S.NF))
            built = build_players(units, self.ref)
            built[:, S.C_OVR] = scout_value(built, self.ref)          # coaches rotate on true contribution, like the front office
            rows[: len(team.roster)] = built
            self._rows[team.id] = rows

    def _sync_avail(self, team) -> None:
        rows = self._rows[team.id]
        rows[:, S.C_AVAIL] = 0.0
        for i, pid in enumerate(team.roster):
            rows[i, S.C_AVAIL] = 0.0 if self.states[pid].injured else 1.0
        if rows[:, S.C_AVAIL].sum() < 7:                     # emergency signing replaces the longest-injured player
            worst = max(team.roster, key=lambda p: self.states[p].games_left)
            self._remove(worst)
            self._new_free_agent(team)
            self.emergency_signings += 1
            self._rebuild_rows()
            self._sync_avail(team)

    def _sim_games(self, pairs, season_res=None, regular=True):
        """Simulate one game per (home, away) pair; apply minutes-driven injuries. Returns scores."""
        G = len(pairs)
        PL = np.zeros((G, 2, S.MAX_ROSTER, S.NF))
        NPL = np.zeros((G, 2), np.int64)
        CO = np.zeros((G, 2, S.NCO))
        for g, (h, a) in enumerate(pairs):
            for side, t in enumerate((h, a)):
                team = self.teams[t]
                self._sync_avail(team)
                PL[g, side] = self._rows[t]
                NPL[g, side] = len(team.roster)
                CO[g, side] = team.coach.row()
        if self.capture_games is not None and regular:
            self.capture_games.append((self.year, PL.copy(), NPL.copy(), CO.copy(), list(pairs)))
        seeds = [self.tree.game_seed(self.year, self._game_id + g + (0 if regular else 100_000)) for g in range(G)]
        self._game_id += G
        BOX, SCORE, POSS = self.eng.simulate_arrays(PL, NPL, CO, seeds)
        out = []
        for g, (h, a) in enumerate(pairs):
            out.append((int(SCORE[g, 0]), int(SCORE[g, 1])))
            rosters = (list(self.teams[h].roster), list(self.teams[a].roster))
            if season_res is not None and regular:
                for side, t in enumerate((h, a)):
                    season_res.team_box[t] += BOX[g, side].sum(0)
                    season_res.opp_box[t] += BOX[g, 1 - side].sum(0)
                    season_res.team_poss[t] += POSS[g, side]
                    season_res.team_games[t] += 1
                    for i, pid in enumerate(rosters[side]):
                        sec = BOX[g, side, i, S.ST_SEC]
                        if sec > 0:
                            season_res.minutes[t][pid] = season_res.minutes[t].get(pid, 0.0) + float(sec)
                            if pid not in season_res.anc:
                                season_res.anc[pid] = self.persons[pid].genome.ancestry.copy()
                season_res.home_margin_sum += float(SCORE[g, 0] - SCORE[g, 1])
                season_res.home_wins += int(SCORE[g, 0] > SCORE[g, 1])
            if self.observers:
                for obs in self.observers:
                    obs.on_game(self, (h, a), rosters, BOX[g], SCORE[g], regular)
            for side, t in enumerate((h, a)):
                self._after_game(self.teams[t], BOX[g, side], season_res, regular)
            if season_res is not None and regular:
                season_res.totals += BOX[g].sum((0, 1))
                season_res.poss += int(POSS[g].sum())
                season_res.n_games += 1
                season_res.seconds += float(BOX[g, :, :, S.ST_SEC].sum())
        return out

    def _after_game(self, team, box, res, regular) -> None:
        """Track lines, tick down injuries for players who sat, then draw new injuries."""
        n = len(team.roster)
        minutes = box[:n, S.ST_SEC] / 60.0
        was_out = np.array([self.states[p].injured for p in team.roster])
        for i, pid in enumerate(team.roster):
            st = self.states[pid]
            if was_out[i]:
                st.games_left -= 1
                if st.games_left == 0:
                    st.injury_type = ""
        avail = ~was_out
        mult = np.array([self.states[p].hazard_mult for p in team.roster])
        ae = np.array([age_eff(self.age(p), self.states[p].race, self.P) for p in team.roster])
        p_inj = np.where(avail, inj.hazard(minutes, mult, self.P), 0.0)
        hit, types, games = inj.draw_injuries(p_inj, ae, self.rng, self.P)
        if res is not None and regular:
            res.injury_exposures += int(avail.sum())
            res.injuries += int(hit.sum())
            res.games_missed.extend(int(g) for g in games if g > 0)
        for pid, ty, g in zip(np.array(team.roster)[hit], types, games):
            if g > 0:
                st = self.states[int(pid)]
                st.injury_type, st.games_left = inj.TYPES[int(ty)], int(g)
        for i, pid in enumerate(team.roster):
            if minutes[i] > 0 and regular:
                self._lines.setdefault(pid, np.zeros(S.NSTAT + 1))
                self._lines[pid][0] += 1
                self._lines[pid][1:] += box[i]

    # ---- season -----------------------------------------------------------------------------
    def _day_schedule(self) -> list[list[tuple[int, int]]]:
        L, n = self.P.league, self.P.league["n_teams"]
        rounds = circle_rounds(n)
        cycles, extra = divmod(L["games"], n - 1)
        days = []
        for c in range(cycles):
            for r in self.rng.permutation(n - 1):
                days.append([(a, b) if (c + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(rounds[r])])
        for e in range(extra):
            r = (self.year + e) % (n - 1)
            days.append([(a, b) if (self.year + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(rounds[r])])
        return days

    def _play_season(self) -> SeasonResult:
        n = self.P.league["n_teams"]
        res = SeasonResult(self.year, np.zeros(n, int), np.zeros(n, int), np.zeros(n))
        res.team_box, res.opp_box = np.zeros((n, S.NSTAT)), np.zeros((n, S.NSTAT))
        res.team_poss, res.team_games = np.zeros(n), np.zeros(n)
        res.minutes = [dict() for _ in range(n)]
        self._lines: dict = {}
        self._game_id = 0
        played = np.zeros(n)
        for day in self._day_schedule():
            for (h, a), (sh, sa) in zip(day, self._sim_games(day, res)):
                w, l = (h, a) if sh > sa else (a, h)
                res.wins[w] += 1
                res.losses[l] += 1
                res.mov[h] += sh - sa
                res.mov[a] += sa - sh
                played[h] += 1
                played[a] += 1
        res.mov = res.mov / np.maximum(played, 1)
        return res

    def _playoffs(self, res: SeasonResult) -> None:
        L = self.P.league
        seeds = rank_teams(res.wins, res.mov)[: L["playoff_teams"]]
        out = run_playoffs(seeds, L["series_length"],
                           lambda h, a: self._sim_games([(h, a)], None, regular=False)[0])
        res.champion, res.finalist = out["champion"], out["finalist"]
        res.series, res.seeds = out["series"], list(seeds)
        self._log("champion", self.teams[res.champion].house_id, (), f"team {res.champion}")

    def _season_end(self, res: SeasonResult) -> None:
        self.last_wins, self.last_mov = res.wins.copy(), res.mov.copy()
        for team in self.teams:
            for i, pid in enumerate(team.roster):
                st = self.states[pid]
                games, mins = (self._lines[pid][0], self._lines[pid][1] / 60.0) if pid in self._lines else (0, 0.0)
                st.log.append((self.year, self.age(pid), float(self._rows[team.id][i, S.C_OVR]), float(games), float(mins)))
