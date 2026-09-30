"""Prospects, house auto-join and the reverse-standings draft."""
from __future__ import annotations

import numpy as np

from dynasty_sim.core.reference import ARCHETYPES, draft_z_score, legacy_cut
from dynasty_sim.core.types import RACES, Person
from dynasty_sim.development.growth import new_state, state_at_age
from dynasty_sim.engine import schema as S
from dynasty_sim.engine.composites import build_players
from dynasty_sim.league.pool import selected_founders
from dynasty_sim.population.commoners import draw_death_year


def is_pro_caliber(league, person: Person) -> bool:
    z = person.genome.A + person.genome.E_perm
    return any(draft_z_score(z, a) >= league.ref.draft_cuts[a] for a in range(len(ARCHETYPES)))


def is_legacy_caliber(league, person: Person) -> bool:
    """Family admission bar: much lower than the open draft's, so houses can field their own (cap-limited)."""
    z = person.genome.A + person.genome.E_perm
    frac = league.P.league["legacy_frac"]
    return any(draft_z_score(z, a) >= legacy_cut(a, frac) for a in range(len(ARCHETYPES)))


def value(league, states: list) -> np.ndarray:
    """Scouting value now plus half the distance to full development (see league/scouting.py)."""
    if not states:
        return np.zeros(0)
    from dynasty_sim.league.scouting import scout_value
    now = scout_value(build_players(np.array([s.units() for s in states]), league.ref), league.ref)
    pot = scout_value(build_players(np.array([s.potential_units() for s in states]), league.ref), league.ref)
    return now + 0.5 * (pot - now)


def generate_prospects(league, n: int, age_lo: int, age_hi: int, rng, g_rng, develop_to_age: bool = False) -> list[int]:
    """Create n commoner prospects (unrelated, selected within their own population).

    Ages are drawn per-race in [adult_age + age_lo, adult_age + age_hi] calendar-scaled years. With
    develop_to_age the state is grown forward from the adult age (used to seed veterans at founding).
    """
    P, pop = league.P, league.pop
    share = np.array([P.league["prospect_race_share"][r] for r in RACES])
    counts = rng.multinomial(n, share / share.sum())
    out = []
    for race, k in enumerate(counts):
        if k == 0:
            continue
        for genome in selected_founders(int(k), RACES[race], g_rng, P, league.ref):
            ts = P.timescale[race]
            age = int(P.adult_age[race] + rng.integers(age_lo, age_hi + 1) * ts)
            pid = pop.ids.next()
            pop.ped.add(pid)
            person = Person(pid, "F" if rng.random() < 0.5 else "M", league.year - age, genome)
            person.death_year = draw_death_year(genome.ancestry, person.birth_year, age, rng, P)
            league.persons[pid] = person
            a0 = int(P.adult_age[race])
            league.states[pid] = (state_at_age(person, a0, age, P, rng, league.ref) if develop_to_age
                                  else new_state(person, age, P, rng, league.ref))
            out.append(pid)
    return out


def eligible_house_members(league, house_id: int, max_ae: float) -> list[int]:
    """Living, pro-caliber house members not currently in the league whose effective age is <= max_ae."""
    from dynasty_sim.development.aging import age_eff
    P, out = league.P, []
    for p in league.pop.alive_people():
        if p.house_id != house_id or p.id in league.states:
            continue
        age = league.year - p.birth_year
        if age < P.adult_age[p.race] or age_eff(age, p.race, P) > max_ae:
            continue
        if is_legacy_caliber(league, p):
            out.append(p.id)
    return out


def sign(league, team, pid: int) -> None:
    team.roster.append(pid)
    st = league.states[pid]
    st.team_id = team.id
    lo, hi = league.P.league["contract_years"]
    st.contract_end = league.year + int(league.rng.integers(lo, hi + 1)) - 1


def house_members_on(league, team) -> int:
    return sum(1 for pid in team.roster if league.persons[pid].house_id == team.house_id)


def league_median_value(league) -> float:
    """Median scouting value of the players currently on rosters (the yardstick for family surplus)."""
    sts = [league.states[p] for t in league.teams for p in t.roster]
    if sts:
        return float(np.median(value(league, sts)))
    return reference_median_value(league)                      # founding: nobody is on a roster yet


def reference_median_value(league) -> float:
    """Median scouting value of the frozen draft-selected reference league (fixed seed; independent of the run)."""
    cached = getattr(league, "_reference_median", None)
    if cached is None:
        from dynasty_sim.league.pool import selected_units
        from dynasty_sim.league.scouting import scout_value
        units = selected_units(3000, np.random.default_rng(20240101), league.P, league.ref)
        cached = league._reference_median = float(np.median(scout_value(build_players(units, league.ref), league.ref)))
    return cached


def family_spent(league, team, ref_value: float) -> float:
    """Talent surplus already committed to a house's family players on its own team."""
    own = [p for p in team.roster if league.persons[p].house_id == team.house_id]
    return float(np.maximum(0.0, value(league, [league.states[p] for p in own]) - ref_value).sum()) if own else 0.0


def house_auto_join(league, max_ae: float = 32.0) -> None:
    """Each house's best pro-caliber members join its own team, up to the house cap."""
    L, P = league.P.league, league.P
    ref_value = league_median_value(league)
    for team in league.teams:
        room = min(L["house_cap"] - house_members_on(league, team), L["roster_max"] - len(team.roster))
        if room <= 0:
            continue
        cands = eligible_house_members(league, team.house_id, max_ae)
        if not cands:
            continue
        for pid in cands:
            p = league.pop.people[pid]
            league.persons[pid] = p
            age = league.year - p.birth_year
            league.states[pid] = state_at_age(p, int(P.adult_age[p.race]), age, P, league.rng, league.ref)
        vals = value(league, [league.states[pid] for pid in cands])
        budget = L["family_budget"] - family_spent(league, team, ref_value)
        signed = 0
        for i in np.argsort(-vals):
            cost = max(0.0, float(vals[i]) - ref_value)
            if signed >= room or cost > budget:
                continue
            sign(league, team, cands[i])
            budget -= cost
            signed += 1
        for pid in cands:                      # unsigned house members go back to the general draft pool
            if league.states[pid].team_id is None:
                league.pool.append(pid)


def run_draft(league, order: list[int], noise: float = 1.5) -> list[tuple[int, int]]:
    """Reverse-standings rounds until every roster is full or the pool is empty. Returns (team, pid) picks."""
    L = league.P.league
    picks = []
    ref_value = league_median_value(league)
    while league.pool:
        progressed = False
        for t in order:
            team = league.teams[t]
            if len(team.roster) >= L["roster_max"] or not league.pool:
                continue
            vals = value(league, [league.states[pid] for pid in league.pool]) + league.rng.normal(0, noise, len(league.pool))
            over = house_members_on(league, team) >= L["house_cap"]      # the cap and the talent budget bind in the draft too
            if not over:
                room = L["family_budget"] - family_spent(league, team, ref_value)
                base = value(league, [league.states[p] for p in league.pool])
                over_budget = np.maximum(0.0, base - ref_value) > room
            else:
                over_budget = np.zeros(len(league.pool), bool)
            if over or over_budget.any():
                own = np.array([league.persons[p].house_id == team.house_id for p in league.pool])
                vals = np.where(own & (over | over_budget), -np.inf, vals)
                if np.isneginf(vals).all():
                    continue
            pid = league.pool.pop(int(np.argmax(vals)))
            sign(league, team, pid)
            picks.append((t, pid))
            progressed = True
        if not progressed:
            break
    return picks


def release_pool(league) -> None:
    """Undrafted players leave: commoner prospects vanish, house members stay in the population."""
    for pid in league.pool:
        league.states.pop(pid, None)
        league.persons.pop(pid, None)
    league.pool = []


def expire_contracts(league) -> int:
    """Contracts ending this offseason. Each team re-signs its best `retain_limit` outsiders and its best
    `family_retain` own-house players (right of first refusal); everyone else re-enters the draft pool."""
    L = league.P.league
    n = 0
    for team in league.teams:
        exp = [p for p in team.roster if league.states[p].contract_end < league.year]
        if not exp:
            continue
        vals = dict(zip(exp, value(league, [league.states[p] for p in exp])))
        fam = sorted((p for p in exp if league.persons[p].house_id == team.house_id), key=lambda p: -vals[p])
        out = sorted((p for p in exp if league.persons[p].house_id != team.house_id), key=lambda p: -vals[p])
        keep = set(fam[: L["family_retain"]]) | set(out[: L["retain_limit"]])
        lo, hi = L["resign_years"]
        for pid in exp:
            st = league.states[pid]
            if pid in keep:
                st.contract_end = league.year + int(league.rng.integers(lo, hi + 1)) - 1
            else:
                team.roster.remove(pid)
                st.team_id = None
                league.pool.append(pid)
                n += 1
    return n


def population_entrants(league, max_ae: float = 23.0) -> None:
    """Any living pro-caliber person in the population (house or commoner lineage) of draft age joins the pool."""
    from dynasty_sim.development.aging import age_eff
    P = league.P
    for p in league.pop.alive_people():
        if p.id in league.states:
            continue
        age = league.year - p.birth_year
        if age < P.adult_age[p.race] or age_eff(age, p.race, P) > max_ae or not is_pro_caliber(league, p):
            continue
        league.persons[p.id] = p
        league.states[p.id] = state_at_age(p, int(P.adult_age[p.race]), age, P, league.rng, league.ref)
        league.pool.append(p.id)
