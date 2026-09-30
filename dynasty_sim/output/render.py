"""Shared renderer: resolves ids to names and links, and narrates events. Every link it emits is recorded, so the
exporter can guarantee each one points at a page that exists."""
from __future__ import annotations

from dynasty_sim.core.types import RACES
from dynasty_sim.history.events import HistoryEvent

RACE_LABEL = {"human": "Human", "goliath": "Goliath", "elf": "Elf"}
STAT_LABEL = {"pts": "points", "reb": "rebounds", "ast": "assists", "stl": "steals", "blk": "blocks", "tpm": "three-pointers",
              "ppg": "points per game", "rpg": "rebounds per game", "apg": "assists per game", "spg": "steals per game",
              "bpg": "blocks per game", "value": "value added", "games": "games played", "team_pts": "team points",
              "team_margin": "winning margin", "wins": "wins", "mov": "average margin"}


class Renderer:
    def __init__(self, world) -> None:
        self.world, self.ledger = world, world.ledger
        self.names = world.ledger.names
        self.need_players: set = set()
        self.need_houses: set = set()
        self.need_games: dict = {}

    # ---- lookups ----------------------------------------------------------------------------------
    def person(self, pid: int):
        return self.world.pop.people.get(pid) or self.ledger.persons.get(pid)

    def race_name(self, pid: int) -> str:
        p = self.person(pid)
        return RACE_LABEL[RACES[p.race]] if p is not None else "Human"

    # ---- links --------------------------------------------------------------------------------------
    def p(self, pid: int, prefix: str = "") -> str:
        self.need_players.add(pid)
        return f"[{self.names.person(pid)}]({prefix}players/{pid}.md)"

    def h(self, hid: int, prefix: str = "") -> str:
        self.need_houses.add(hid)
        return f"[{self.names.house(hid)}]({prefix}houses/{hid}.md)"

    def t(self, team_id: int) -> str:
        return self.names.team(team_id)

    def team_house(self, team_id: int, prefix: str = "") -> str:
        return f"{self.t(team_id)} ({self.h(self.world.league.teams[team_id].house_id, prefix)})"

    # ---- narration -----------------------------------------------------------------------------------
    def narrate(self, e: HistoryEvent, prefix: str = "") -> str:
        P = lambda pid: self.p(pid, prefix)
        H = lambda hid: self.h(hid, prefix)
        d, k = e.detail, e.kind
        who = ", ".join(P(a) for a in e.actors)
        if k == "award":
            return f"{who} was named {d['award']} ({e.value:.1f} points added)."
        if k == "record":
            stat = STAT_LABEL.get(d["stat"], d["stat"])
            scope = "league" if d["scope"] == "all" else RACE_LABEL[d["scope"]]
            what = {"game": "single-game", "season": "single-season", "career": "career",
                    "team_game": "single-game team", "team_season": "single-season team"}[d["record_kind"]]
            if e.actors:
                holder, old = who, P(d["old_holder"]) if d["record_kind"] != "team_game" else ""
            else:
                holder, old = self.t(d["team"] if "team" in d else self._team_of_house(e.houses[0])), ""
            edge = "lowest" if d["direction"] == "min" else "record"
            prev = f" (previous {d['old_value']:g}" + (f" by {old}" if old else "") + f", {d['old_year']})"
            return f"{holder} set a new {scope} {what} {stat} {edge}: {e.value:g}{prev}."
        if k == "title":
            core = f" led by {who}" if who else ""
            return f"{self.t(d['team'])} ({H(e.houses[0])}) won the title{core}."
        if k == "drought_ended":
            return f"{H(e.houses[0])} ended a title drought of {e.value:.0f} years."
        if k == "upset":
            return (f"{self.t(d['winner_team'])} upset {self.t(d['loser_team'])} in a playoff series the model gave them "
                    f"a {100 * e.value:.0f}% chance of winning" + (f", starring {who}." if who else "."))
        if k == "prodigy":
            return f"{who} broke through as a {'hybrid ' if d['hybrid'] else ''}prodigy, among the league's five best before turning 23."
        if k == "trait_exposed":
            return f"{who} showed a rare congenital trait ({', '.join(d['traits'])}), exposed by inheriting it from both parents."
        if k == "outlier":
            if e.actors:
                return f"{who} posted one of the best seasons on record (higher than {100 * e.percentile:.1f}% of all before)."
            return f"{H(e.houses[0])} fielded a historically dominant team (margin above {100 * e.percentile:.1f}% of all seasons before)."
        if k == "rivalry":
            a, b = e.houses
            why = " and ".join(d.get("why", [])) or "repeated meetings"
            return f"{H(a)} and {H(b)} became rivals, fuelled by {why} (rivalry index {e.value:.0f})."
        if k in ("retire", "death_in_service"):
            what = "retired" if k == "retire" else "died while still playing"
            return f"{who} {what} after a career worth {e.value:.0f} points added."
        if k == "succession":
            dead, heir = e.actors
            return f"{P(dead)}, head of {H(e.houses[0])}, died; {P(heir)} succeeded as head ({d['how']})."
        if k == "adoption":
            dead, heir = e.actors
            return f"{H(e.houses[0])} had no heir after {P(dead)}; {P(heir)} was adopted to carry on the house ({d['how']})."
        if k == "extinction":
            return f"{H(e.houses[0])} died out with {who}."
        if k == "marriage":
            a, b = e.houses
            return f"{P(e.actors[0])} of {H(a)} married {P(e.actors[1])} of {H(b)}."
        if k == "heir_death":
            return f"{who}, heir to {H(e.houses[0])}, died young ({d['age']})."
        if k == "young_death":
            return f"{who} of {H(e.houses[0])} died young ({d['age']})."
        if k == "hof":
            return f"{who} was inducted into the Hall of Fame (career value {e.value:.0f})."
        if k == "hof_builder":
            return f"{who} was honoured as a dynasty builder for {e.value:.0f} titles as head of {H(e.houses[0])}."
        if k == "succession_effect":
            verb = "improved" if e.value > 0 else "fell"
            return (f"Three seasons after {P(e.actors[0])} took over {H(e.houses[0])}, the team's win rate {verb} by "
                    f"{abs(e.value) * 100:.0f} points ({d['before']:.0%} to {d['after']:.0%}).")
        return f"{k.replace('_', ' ')}: {who}".strip()

    @staticmethod
    def fy(y: int) -> str:
        """Long form of a year; founders were born before year 0."""
        return f"year {y}" if y >= 0 else f"{-y} years before the founding"

    @staticmethod
    def fys(y: int) -> str:
        return str(y) if y >= 0 else f"{-y} BF"

    def _team_of_house(self, hid: int) -> int:
        return next(t.id for t in self.world.league.teams if t.house_id == hid)
