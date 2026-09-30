# Dynasty Basketball Sim (proof of concept)

A generational basketball simulation. Families ("houses") breed players over decades; three races (Human, Goliath,
Elf) differ in physical traits, not in bonuses; a possession-level engine turns those traits into games; and the
whole history (careers, records, rivalries, succession dramas) is written out as readable Markdown.

The design and all research behind it are in
[Research Reference and Build Spec for a Proof of Concept.md](Research%20Reference%20and%20Build%20Spec%20for%20a%20Proof%20of%20Concept.md).
Where this build departs from that spec, the reason is written next to the setting (see [Deviations](#deviations-from-the-spec)).

## Install

Requires Python 3.11 or newer (numba is the constraint; 3.11 is the tested version).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Windows
# .venv/bin/python -m pip install -e ".[dev]"        # macOS / Linux
```

The first run compiles the numba engine, which takes about a minute. The result is cached, so later runs start fast.

## Run a world

```bash
.venv/Scripts/python -m dynasty_sim.cli run --years 50 --seed 3 --out runs/demo
```

| Option | Meaning |
|---|---|
| `--years N` | seasons to simulate (default 1) |
| `--seed N` | run seed; the same seed always gives the same history (default 42) |
| `--teams N` | teams in the league (default 8; one per house) |
| `--out DIR` | output folder (default `runs`) |
| `--parquet` | also write `player_seasons`, `events` and `team_seasons` Parquet tables |
| `--no-history` | skip the Markdown pages (faster; still writes the JSON files) |

Timing on a 4-core machine, including the history export: about 33 s for 50 years, about 64 s for 100 years.

### What you get

```
runs/demo/
  manifest.json       seed, config hashes, champions, alarms, counts of pages/events/arcs
  dashboard.json      per-decade trait means and 99th percentiles, team margins, title Gini
  reference.json      the frozen year-0 reference every z-score is measured against
  people.json         every person: parents, house, inbreeding coefficient, ancestry, traits
  history/
    index.md          start here
    history.md        season by season: champions, MVP, awards, notable events
    records.md        game, season and career records, per race and overall
    hall_of_fame.md   inductees and dynasty builders
    rivalries.md      the house rivalries and why they formed
    arcs.md           storylines: a person's or house's chain of linked events
    players/<id>.md   one page per player: career table, honours, family, timeline
    houses/<n>.md     one page per house: titles, successions, family tree
    houses/<n>.dot    the same family tree for Graphviz
    games/            box scores of title-deciding and record games
```

Every link in the Markdown resolves to a file that exists; the exporter checks this and reports any that do not in
`manifest.json` under `history.unresolved`.

To draw a family tree: `dot -Tsvg runs/demo/history/houses/7.dot -o house7.svg` (needs [Graphviz](https://graphviz.org)).

## Run many worlds at once

```bash
.venv/Scripts/python -m dynasty_sim.cli sweep --seeds 200 220 --years 100
```

Runs seeds 200 to 219 in parallel and reports the stability dashboards and alarms (trait drift, runaway margins, a
single house winning over 40% of titles in 30 seasons). Add `--workers N` to change the process count.

## Tests

```bash
.venv/Scripts/python -m pytest                    # everything, about 15 minutes
.venv/Scripts/python -m pytest -m "not slow"      # skip the long statistical runs
.venv/Scripts/python -m pytest -m acceptance      # the spec's acceptance tests (a)-(e), about 8 minutes
```

| Milestone | Tests |
|---|---|
| M0-M1 | seeded reproducibility; pedigree F, heritability slope, variance over generations |
| M2 | population and house survival across 50 seeds; succession order |
| M3-M4 | engine invariants and speed; calibration against the validation table |
| M5 | peak ages, injury rates, roster and draft rules |
| M6 | stability mechanisms, alarms, 100-year runs over 20 seeds |
| M7 | history events, arcs, records, awards, Markdown output |
| M8 | acceptance tests (a) styles, (b) stability, (c) hybrid variance, (d) retellable history, (e) performance |

## How the code is laid out

```
dynasty_sim/
  config/        defaults.yaml, races.yaml, races_balance.yaml, engine*.yaml, validation.yaml, scouting.yaml
  core/          seeded RNG tree, types, the frozen reference
  genetics/      genome, inheritance, pedigree (inbreeding), Mendelian loci, hybrids
  population/    demography, succession, commoners
  development/   aging, skill learning, injuries, player state
  engine/        composites, the numba kernel, calibration
  league/        season loop, draft, scouting, playoffs, world runner
  history/       ledger, records, awards, Hall of Fame, rivalries, arcs, dashboards, alarms
  output/        Markdown pages, box scores, family trees, export
  cli.py         command line
tests/
```

Two ideas run through everything:

- **The engine never sees a race.** Game inputs are physical traits in real units; styles come out of the physics, and a
  test confirms that shuffling race labels against traits erases every style difference.
- **Determinism.** Each subsystem draws from its own seeded stream, so recording history or changing narrative code can
  never alter a game result. Two runs with the same seed produce byte-identical output.

## Configuration

Everything tunable is in `dynasty_sim/config/`. The ones you are most likely to want:

| File / key | What it does |
|---|---|
| `defaults.yaml` `demography.K` | carrying capacity per race |
| `defaults.yaml` `demography.house_races` | the race of each founding house (one per team) |
| `defaults.yaml` `league.prospect_race_share` | mix of the yearly commoner draft prospects |
| `defaults.yaml` `league.house_cap`, `family_budget` | how many family players a house can field, and how much talent |
| `defaults.yaml` `demography.kin_avoid_prob`, `mate_choice_lambda` | how often relatives marry; preference for athletic mates |
| `defaults.yaml` `league.coach_effect` | how much a house head's traits shape the team's coaching style (0 = none) |
| `races.yaml` / `races_balance.yaml` | race trait tables, and the recorded departures from them |

To test a variant without editing files, the parallel runner accepts overrides, for example
`sweep(range(20), 100, {"league.family_budget": 12.0})` from `dynasty_sim.history.sweep`.

### Recalibrating the engine

The engine's probabilities are fitted so a Human league lands inside the validation table. If you change anything that
alters how games or rosters come out, refit and freeze the result:

```python
from dynasty_sim.core.reference import frozen_reference
from dynasty_sim.engine.calibrate import calibrate, write_calibrated, STAGE2
from dynasty_sim.engine.interface import NumbaEngine
from dynasty_sim.league.calsets import build_loop_league_set

ref, eng, ls = frozen_reference(), NumbaEngine(), build_loop_league_set(2024)
params, loss = calibrate(eng, ls, STAGE2, max_evals=2500)
write_calibrated(params, loss, ref)
```

This takes about ten minutes. The calibration league is the Human-only league the simulation itself produces, and its
rosters depend slightly on the current calibration, so repeat until a freshly built league passes without a refit.
Do not run two calibrations at once or run other work that reads the config while one is writing it: the result is
written to `config/engine_calibrated.yaml`, and `tests/test_m4_calibration.py` fails if that file is stale.

## Deviations from the spec

Each is recorded where it lives; the short list:

- **Elf trait table:** height mean 178 (spec 189) and touch 68 (spec 62), in `races_balance.yaml`.
- **Hybrid variance:** the Castle-Wright factor is 4 x ancestry-product (the spec's 2 is half the textbook value), `n_eff`
  is 6, and the F1 "developmental instability" override skips height and strength.
- **Skill ceiling** is the genetic value, not genetic + 1.5, so the calibrated baseline is not inflated.
- **Injury height penalty** is measured within a person's own race, not against Humans.
- **Rivalry threshold** is 1.07x the median pairing for three straight seasons, because the spec's point weights put every
  pairing at nearly the same level.
- **Validation clause** "Human-majority seasons inside the table" is read per table row (average at least 90%), since the
  table's own midpoints cannot all hold at once.
- **Hybrid fertility** is milder than the spec default: `hybrid_fert_k` 1.0 (spec 2.0, allowed 1-3) and `p_cross_race` 0.15
  (spec 0.08). At the spec values only 0.1% of people were hybrids; now about 0.7%, with F2 lines and the occasional star.
- **Scoring realism:** shooting skill has diminishing returns (`touch_sat`), heavy usage costs efficiency (`usage_cost`),
  and free throws are drawn by strength (post play, saturating) and by handling on drives (`sf_drive`, twos only).
  Late in close games the defence fouls the worst free-throw shooter on the floor (`hack_p`, `hack_secs`, `hack_margin`).
- **Hall of Fame** scores a retired career as the mean of three standardised parts: career value, best five seasons, and
  honours (awards and titles). Longevity alone does not carry a player.
- **Scouting** uses a value fitted to simulated winning contribution, not the engine's game rating.

## Known limits

- Hall of Fame race mix follows the strength of each race's best players, so Goliaths and Elves are over-represented
  relative to their share of eligible careers.
- Title concentration (one house over 40% of titles in 30 seasons) still trips in about 5% of seeds.
- The Elf effect on three-point rate is positive but only barely resolved statistically.
- The validation table's strictest reading (every row inside in the same season) holds in only 30-40% of seasons.
- Family-tree inbreeding is rare by default (kin avoidance is on); raise `kin_avoid_prob` toward 0 to see more of it.
