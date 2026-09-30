"""Ledger: collects everything worth retelling while a world runs (careers, records, awards, HoF, rivalries, events)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dynasty_sim.core.types import LOCI, RACES
from dynasty_sim.development.aging import age_eff
from dynasty_sim.engine import schema as S
from dynasty_sim.history import awards as AW
from dynasty_sim.history import events as EV
from dynasty_sim.history.hof import HallOfFame
from dynasty_sim.history.names import NameBook
from dynasty_sim.history.records import (
    MIN_GAMES_FOR_SEASON_RECORD, PLAYER_CAREER, PLAYER_GAME, PLAYER_SEASON, RecordBook,
)
from dynasty_sim.history.rivalry import RivalryBoard, pair

OUTLIER_PCT = 0.995
UPSET_PROB = 0.15
DROUGHT_YEARS = 20
CONSEQUENCE = 0.15
_HOMOZYGOUS_TRAITS = ("titan", "brittle", "stonehands")


@dataclass
class SeasonLine:
    year: int
    age: int
    team: int
    games: int
    stats: np.ndarray
    value: float = 0.0


@dataclass
class Career:
    pid: int
    sex: str
    race: int
    ancestry: np.ndarray
    birth_year: int
    mother_id: Optional[int]
    father_id: Optional[int]
    hyb_gen: int
    seasons: list = field(default_factory=list)
    houses: list = field(default_factory=list)          # house id of each season's team
    retired_year: Optional[int] = None
    hof_year: Optional[int] = None
    awards: list = field(default_factory=list)           # (year, award)
    exposed: list = field(default_factory=list)          # homozygous congenital traits

    def career_value(self) -> float:
        return float(sum(s.value for s in self.seasons))

    def total(self, k: int) -> float:
        return float(sum(s.stats[k] for s in self.seasons))

    def games(self) -> int:
        return int(sum(s.games for s in self.seasons))


@dataclass
class GameSnapshot:
    year: int
    teams: tuple
    score: tuple
    rosters: tuple
    box: np.ndarray
    playoff: bool


class Ledger:
    def __init__(self, world) -> None:
        self.world = world
        self.log = EV.EventLog()
        self.careers: dict[int, Career] = {}
        self.persons: dict[int, object] = {}
        self.records = RecordBook()
        self.hof = HallOfFame()
        self.rivalry = RivalryBoard()
        self.awards: dict[int, dict] = {}
        self.values: dict[int, dict] = {}                  # year -> {pid: season value}
        self.snapshots: list[GameSnapshot] = []
        self.rivalry_top: dict[int, list] = {}
        self.names = NameBook(world.seed, lambda pid: world.pop.people.get(pid) or self.persons.get(pid))
        self.titles: dict[int, list] = {}                   # house -> [years]
        self._pop_ptr = 0
        self._value_hist: list = []
        self._mov_hist: list = []
        self._heads: dict[int, tuple] = {}                 # house -> (pid, start year)
        self._successions: list = []                       # (year, house, dead head, heir, event id)
        self._season_team: dict = {}                        # pid -> team id at his most recent game this season
        self._best_game: tuple = (-1, None)
        self._season_games: list = []
        self.succession_effects: list = []                 # dicts with year, house, delta, flagged

    # ---- helpers -------------------------------------------------------------------------------------
    def house_of_team(self, team_id: int) -> int:
        return self.world.league.teams[team_id].house_id

    def _career(self, lg, pid: int) -> Career:
        c = self.careers.get(pid)
        if c is None:
            p = self.persons.get(pid) or lg.persons[pid]
            self.persons[pid] = p
            g = p.genome
            copies = g.loci.sum(axis=1)
            exposed = [n for n in _HOMOZYGOUS_TRAITS if copies[LOCI.index(n)] == 2]
            c = self.careers[pid] = Career(pid, p.sex, p.race, g.ancestry.copy(), p.birth_year, p.mother_id, p.father_id,
                                           g.hyb_gen, exposed=exposed)
        return c

    # ---- per-game hook ---------------------------------------------------------------------------------
    def on_game(self, lg, pair_, rosters, box, score, regular: bool) -> None:
        h, a = pair_
        hh, ha = lg.teams[h].house_id, lg.teams[a].house_id
        year = lg.year
        for side, roster in enumerate(rosters):                 # players can be released mid-season; remember who they were
            for pid in roster:
                if pid not in self.persons and pid in lg.persons:
                    self.persons[pid] = lg.persons[pid]
                if regular:
                    self._season_team[pid] = pair_[side]
        if regular:
            self.rivalry.game(hh, ha, int(score[0] - score[1]))
            hi = self._record_game(lg, year, pair_, rosters, box, score)
        else:
            self.snapshots.append(GameSnapshot(year, pair_, (int(score[0]), int(score[1])), (tuple(rosters[0]), tuple(rosters[1])),
                                               box.copy(), True))

    def _record_game(self, lg, year, pair_, rosters, box, score) -> None:
        R = self.records
        for side in (0, 1):
            R.offer("team_game", "team_pts", int(score[side]), pair_[side], year, direction="max")
            R.offer("team_game", "team_pts", int(score[side]), pair_[side], year, direction="min")
            R.offer("team_game", "team_margin", int(score[side] - score[1 - side]), pair_[side], year)
            for i, pid in enumerate(rosters[side]):
                row = box[side, i]
                if row[S.ST_SEC] <= 0:
                    continue
                vals = {"pts": row[S.ST_PTS], "reb": row[S.ST_ORB] + row[S.ST_DRB], "ast": row[S.ST_AST],
                        "stl": row[S.ST_STL], "blk": row[S.ST_BLK], "tpm": row[S.ST_TPM]}
                race = RACES[lg.persons[pid].race] if pid in lg.persons else "human"
                for stat, v in vals.items():
                    R.offer("game", stat, v, pid, year, ("all", race))
                if vals["pts"] > self._best_game[0]:
                    self._best_game = (vals["pts"], GameSnapshot(year, pair_, (int(score[0]), int(score[1])),
                                                                 (tuple(rosters[0]), tuple(rosters[1])), box.copy(), False))

    # ---- per-season processing ---------------------------------------------------------------------------
    def after_season(self, world, res) -> None:
        lg, year = world.league, world.year
        self._consume_population_events(lg, year)
        lines, team_of = self._season_lines(lg, year)
        values = AW.season_values(lines)
        self.values[year] = values
        self._store_lines(lg, year, lines, team_of, values)
        self._awards(lg, year, res, lines, team_of, values)
        self._season_records(lg, year, res, lines, values)
        self._title_and_series(lg, year, res, values, team_of)
        self._prodigies_and_traits(lg, year, values)
        self._outliers(lg, year, res, values)
        self._moves(lg, year, team_of)
        self._rivalry_flags(year)
        self._heads_and_hof(world, year)
        self._succession_effects(lg, year)
        if self._best_game[1] is not None:
            self.snapshots.append(self._best_game[1])
        self._best_game = (-1, None)

    def _season_lines(self, lg, year):
        lines, team_of = {}, {}
        for pid, v in lg._lines.items():                        # everyone who played, including players since released
            if v[0] > 0 and pid in self._season_team:
                lines[pid] = v
                team_of[pid] = self._season_team[pid]
        self._season_team = {}
        return lines, team_of

    def _store_lines(self, lg, year, lines, team_of, values) -> None:
        for pid, v in lines.items():
            c = self._career(lg, pid)
            c.seasons.append(SeasonLine(year, year - c.birth_year, team_of[pid], int(v[0]), v[1:].copy(), float(values[pid])))
            c.houses.append(self.house_of_team(team_of[pid]))

    def _awards(self, lg, year, res, lines, team_of, values) -> None:
        prev = self.values.get(year - 1, {})
        rookies = {p for p in lines if len(self.careers[p].seasons) == 1}
        aw = AW.yearly_awards(lines, values, team_of, rookies, prev, res.champion)
        self.awards[year] = aw
        spec = {"mvp": ("MVP", EV.LANDMARK), "finals_mvp": ("Finals MVP", EV.MAJOR), "dpoy": ("Defensive Player", EV.MAJOR),
                "roy": ("Rookie of the Year", EV.MAJOR), "scoring": ("Scoring Champion", EV.NOTABLE),
                "most_improved": ("Most Improved", EV.NOTABLE)}
        for key, (label, imp) in spec.items():
            pid = aw.get(key)
            if pid is None:
                continue
            self.careers[pid].awards.append((year, label))
            self.log.add("award", year, (pid,), (self.house_of_team(team_of[pid]),), imp, "season_value", values[pid],
                         award=label)
        for pid in aw.get("all_league", []):
            self.careers[pid].awards.append((year, "All-League"))
            self.log.add("award", year, (pid,), (self.house_of_team(team_of[pid]),), EV.NOTABLE, "season_value",
                         values[pid], award="All-League")

    def _season_records(self, lg, year, res, lines, values) -> None:
        R = self.records
        for pid, v in lines.items():
            c = self.careers[pid]
            race = RACES[c.race]
            if v[0] >= MIN_GAMES_FOR_SEASON_RECORD:
                pg = AW.per_game(v)
                for stat in PLAYER_SEASON:
                    R.offer("season", stat, values[pid] if stat == "value" else pg[stat], pid, year, ("all", race))
            tot = {"pts": c.total(S.ST_PTS), "reb": c.total(S.ST_ORB) + c.total(S.ST_DRB), "ast": c.total(S.ST_AST),
                   "stl": c.total(S.ST_STL), "blk": c.total(S.ST_BLK), "games": c.games(), "value": c.career_value()}
            for stat in PLAYER_CAREER:
                R.offer("career", stat, tot[stat], pid, year, ("all", race))
        for t in range(len(res.wins)):
            R.offer("team_season", "wins", int(res.wins[t]), t, year, direction="max")
            R.offer("team_season", "wins", int(res.wins[t]), t, year, direction="min")
            R.offer("team_season", "mov", float(res.mov[t]), t, year, direction="max")
            R.offer("team_season", "mov", float(res.mov[t]), t, year, direction="min")
        for b in R.finalize(year):
            team_rec = b.kind.startswith("team")
            holder_house = self.house_of_team(b.new.holder) if team_rec else self._house_of_pid(b.new.holder)
            headline = b.stat in ("pts", "ppg", "value", "wins", "mov", "team_margin")
            imp = EV.MINOR if b.scope != "all" else (EV.MAJOR if headline and b.kind in ("season", "career", "team_season", "game") else
                                                     EV.NOTABLE if b.kind in ("season", "career", "team_season") or headline else EV.MINOR)
            self.log.add("record", year, () if team_rec else (b.new.holder,), (holder_house,), imp, f"{b.kind}:{b.stat}",
                         b.new.value, scope=b.scope, direction=b.direction, old_value=b.old.value, old_holder=b.old.holder,
                         old_year=b.old.year, stat=b.stat, record_kind=b.kind)

    def _house_of_pid(self, pid: int) -> int:
        c = self.careers.get(pid)
        return c.houses[-1] if c and c.houses else 0

    def _title_and_series(self, lg, year, res, values, team_of) -> None:
        champ_house = self.house_of_team(res.champion)
        core = sorted((p for p, t in team_of.items() if t == res.champion), key=lambda p: -values[p])[:3]
        last = max(self.titles.get(champ_house, [-999]))
        seen_before = year > DROUGHT_YEARS and (year - last > DROUGHT_YEARS or last == -999)
        self.titles.setdefault(champ_house, []).append(year)
        self.log.add("title", year, tuple(core), (champ_house,), EV.MAJOR, "wins", float(res.wins[res.champion]),
                     team=res.champion)
        if seen_before:
            self.log.add("drought_ended", year, tuple(core), (champ_house,), EV.LANDMARK, "years",
                         float(year - last) if last != -999 else float(year), team=res.champion)
        for hi, lo, winner, rnd in res.series:
            hh, hl = self.house_of_team(hi), self.house_of_team(lo)
            self.rivalry.playoff_series(hh, hl)
            p_hi_game = 0.5 * (EV.game_win_prob(res.mov[hi], res.mov[lo]) + 1 - EV.game_win_prob(res.mov[lo], res.mov[hi]))
            p_hi = EV.series_win_prob(p_hi_game, lg.P.league["series_length"])
            if winner == lo and 1 - p_hi < UPSET_PROB:
                star = max((p for p, t in team_of.items() if t == lo), key=lambda p: values[p], default=None)
                self.log.add("upset", year, (star,) if star is not None else (), (hh, hl), EV.MAJOR, "series_win_prob",
                             1 - p_hi, winner_team=lo, loser_team=hi, round=rnd)

    def _prodigies_and_traits(self, lg, year, values) -> None:
        ranked = sorted(values, key=lambda p: -values[p])[:5]
        for pid in ranked:
            c = self.careers[pid]
            if age_eff(year - c.birth_year, c.race, lg.P) < 23:
                hybrid = c.hyb_gen >= 1
                self.log.add("prodigy", year, (pid,), (c.houses[-1],), EV.MAJOR if hybrid else EV.NOTABLE,
                             "season_value", values[pid], hybrid=hybrid)
        for pid in values:
            c = self.careers[pid]
            if len(c.seasons) == 1 and c.exposed:
                self.log.add("trait_exposed", year, (pid,), (c.houses[-1],), EV.NOTABLE, "", None, traits=list(c.exposed))

    def _outliers(self, lg, year, res, values) -> None:
        qual = [v for p, v in values.items() if lg._lines[p][0] >= MIN_GAMES_FOR_SEASON_RECORD]
        if len(self._value_hist) >= 400:
            for pid, v in values.items():
                if lg._lines[pid][0] >= MIN_GAMES_FOR_SEASON_RECORD:
                    pct = EV.percentile_rank(v, self._value_hist)
                    if pct >= OUTLIER_PCT:
                        self.log.add("outlier", year, (pid,), (self._house_of_pid(pid),), EV.MAJOR, "season_value", v, pct)
        self._value_hist += qual
        if len(self._mov_hist) >= 40:
            for t, m in enumerate(res.mov):
                pct = EV.percentile_rank(float(m), self._mov_hist)
                if pct >= OUTLIER_PCT:
                    self.log.add("outlier", year, (), (self.house_of_team(t),), EV.MAJOR, "team_mov", float(m), pct, team=t)
        self._mov_hist += [float(m) for m in res.mov]

    def _moves(self, lg, year, team_of) -> None:
        for pid, t in team_of.items():
            seasons = self.careers[pid].seasons
            if len(seasons) >= 2 and seasons[-2].team != seasons[-1].team:
                self.rivalry.player_move(self.house_of_team(seasons[-2].team), self.house_of_team(t))

    def _rivalry_flags(self, year: int) -> None:
        for a, b in self.rivalry.end_season(year):
            self.log.add("rivalry", year, (), (a, b), EV.NOTABLE, "rivalry_index", float(self.rivalry.index[(a, b)]),
                         why=self.rivalry.why((a, b)))
        self.rivalry_top[year] = self.rivalry.top(3)

    # ---- population events --------------------------------------------------------------------------------
    def _consume_population_events(self, lg, year: int) -> None:
        evs = lg.pop.events
        for e in evs[self._pop_ptr:]:
            if e.kind in ("retire", "player_death"):
                c = self.careers.get(e.actors[0])
                if c is not None and c.retired_year is None:
                    c.retired_year = e.year
                    star = len(c.seasons) >= 6 and c.career_value() >= self._retire_bar()
                    self.log.add("retire" if e.kind == "retire" else "death_in_service", e.year, e.actors, (e.house_id,) if e.house_id else (),
                                 EV.NOTABLE if star or e.kind == "player_death" else EV.MINOR, "career_value", c.career_value())
            elif e.kind in ("succession", "adoption"):
                ev = self.log.add("succession" if e.kind == "succession" else "adoption", e.year, e.actors, (e.house_id,),
                                  EV.MAJOR, how=e.detail)
                self._successions.append((e.year, e.house_id, e.actors[0], e.actors[1], ev.id))
                if e.kind == "adoption" and "from house" in e.detail:
                    origin = e.detail.rsplit(" ", 1)[-1]
                    if origin.isdigit():
                        self.rivalry.marriage_or_dispute(e.house_id, int(origin), "disputes")
            elif e.kind == "extinction":
                self.log.add("extinction", e.year, e.actors, (e.house_id,), EV.LANDMARK)
            elif e.kind == "marriage":
                hw, hm = (int(x) for x in e.detail.split("-"))
                self.rivalry.marriage_or_dispute(hw, hm, "marriages")
                self.log.add("marriage", e.year, e.actors, (hw, hm), EV.NOTABLE)
            elif e.kind == "heir_death":
                self.log.add("heir_death", e.year, e.actors, (e.house_id,), EV.MAJOR, age=e.detail)
            elif e.kind == "young_death":
                self.log.add("young_death", e.year, e.actors, (e.house_id,), EV.MINOR)
        self._pop_ptr = len(evs)

    def _retire_bar(self) -> float:
        vals = [c.career_value() for c in self.careers.values() if c.retired_year is not None and len(c.seasons) >= 5]
        return float(np.quantile(vals, 0.8)) if len(vals) >= 15 else float("inf")

    # ---- heads, hall of fame, succession consequences ---------------------------------------------------------
    def _heads_and_hof(self, world, year: int) -> None:
        ended = []
        for hid, house in world.pop.houses.items():
            cur = self._heads.get(hid)
            if cur is None or cur[0] != house.head_id:
                if cur is not None:
                    n = sum(1 for y in self.titles.get(hid, []) if cur[1] <= y <= year)
                    ended.append((cur[0], hid, cur[1], year, n))
                self._heads[hid] = (house.head_id, year) if house.head_id is not None else None
                if self._heads[hid] is None:
                    del self._heads[hid]
        for pid in self.hof.inductees_this_year(year, self.careers):
            c = self.careers[pid]
            c.hof_year = year
            c.awards.append((year, "Hall of Fame"))
            self.log.add("hof", year, (pid,), (c.houses[-1],), EV.LANDMARK, "career_value", c.career_value())
        for pid, hid, titles in self.hof.builders_this_year(year, ended):
            self.log.add("hof_builder", year, (pid,), (hid,), EV.LANDMARK, "titles", float(titles))

    def _succession_effects(self, lg, year: int) -> None:
        hist = lg.history
        for (y, hid, dead, heir, evid) in list(self._successions):
            if year == y + 2 and y >= 2:                                     # three seasons played under the new head
                t = hid - 1
                win = lambda r: r.wins[t] / max(r.wins[t] + r.losses[t], 1)
                before, after = win(hist[y - 2]), float(np.mean([win(hist[k]) for k in range(y - 1, y + 2)]))
                delta = after - before
                flagged = abs(delta) > CONSEQUENCE
                self.succession_effects.append({"year": y, "house": hid, "delta": delta, "flagged": flagged, "heir": heir})
                self.log.add("succession_effect", year, (heir,), (hid,), EV.MAJOR if flagged else EV.MINOR, "win_pct_change",
                             delta, flagged=flagged, succession_year=y, before=before, after=after)
                self._successions.remove((y, hid, dead, heir, evid))
