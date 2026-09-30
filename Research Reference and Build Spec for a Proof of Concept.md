# Dynasty Basketball Simulation: Research Reference and Build Spec for a Proof of Concept

The PoC is buildable as specified. Use an additive "infinitesimal" genetics model (offspring breeding value = midparent + Gaussian segregation noise shrunk by parental inbreeding) with a small Mendelian layer for rare, legible congenital traits. Feed physical traits in absolute units (cm, seconds) into a numba-compiled logistic possession engine, calibrated once against a frozen reference league so race playstyles emerge from physics, not bonuses. The biggest risks are stability risks, not engine risks: runaway selective breeding, a closed dynasty gene pool with no outside anchor, and family teams monopolizing their own bloodline. Each needs a counter-mechanism from day one.

## TL;DR

- **Genetics:** store one additive breeding value per trait (phenotypic-SD units) plus race-ancestry fractions and a few Mendelian loci. Child value = midparent + N(0, (h²/2)·(1 − (F_mother + F_father)/2)). Compute F with the tabular coancestry method. Anchor inbreeding depression on Joshi et al. 2015 (Nature; 354,224 people in 102 cohorts: first-cousin offspring 1.2 cm shorter) and Bittles & Black 2010 (PNAS; 3.5% excess pre-reproductive mortality at first-cousin level across 69 study populations).
- **Engine:** per-possession logistic model where turnover, shot type, contest, make, block, foul and rebound are logit shifts driven by trait *differences* (reach − release height, quickness gaps, rebounding mass/reach). Calibrate to NBA targets: pace ≈ 99–100, ORtg ≈ 110–115, eFG% ≈ .52–.55, TOV% ≈ 12.8–13.6, FTA/FGA ≈ .20–.24, home edge ≈ 2–3 points. About 1.2M possessions for 50 years × 8 teams should run in seconds under numba.
- **Stability:** anchor the league with a stationary "commoner" draft pool from fixed race distributions, freeze a year-0 reference population to measure drift, cap family members on the family's own team, and make extremes costly through physiological trade-offs rather than rating caps. Copy ZenGM's fixed age curves plus regenerated draft classes, but note how fragile this is: in the developer's own 2023 test, a coaching-spending bug moved 70+ OVR players from 7.9 to 8.9 and top-team OVR from 75.5 to 82.5 over 20 simmed seasons (ZenGM blog, Aug 2023).

## Executive Summary

**Best-grounded inputs:** height h² ≈ 0.80; endurance ≈ 0.50; inbreeding costs 1.2 cm at first-cousin level with no human fertility deficit; Basketball GM's open-source possession loop as the engine template (details and sources in §1–§3).

**Prioritized design decisions (most important first)**

1. **Traits in physical units drive the engine; the engine never sees race.** GameInput has no race field. This alone makes test (a) pass honestly.
2. **Freeze a year-0 reference population** for z-scoring and drift metrics. Never re-normalize against the current league, or inflation becomes invisible (OOTP's "ratings relative to league" is display-only for this reason).
3. **A stationary commoner draft pool is the gene-flow anchor.** Dynasty-bred players should fill at most about 40–60% of roster slots at equilibrium [INFERRED].
4. **Selection limits come from physics and trade-offs, not caps:** height raises injury hazard and lowers lateral speed; mass raises fatigue; add viability selection on extremes and a pleiotropic correlation matrix.
5. **Hybrid genetics model:** additive infinitesimal values for 12 continuous traits plus 4–6 Mendelian loci for rare congenital traits, exposed by inbreeding through real identity-by-descent.
6. **Model hybrid variance explicitly:** an ancestry-heterozygosity term widens F2+; a labeled design override widens F1, because biology says F1s are usually *not* more variable.
7. **One critical learning window per race,** scaled by a race timescale.
8. **Separate numba kernel from Python orchestration:** genetics/demography in plain Python (hundreds of events/year), possessions in numba (millions).
9. **Reproducibility:** numpy `SeedSequence.spawn` per subsystem and season, so narrative code changes never alter game results.
10. **Data-only engine interface** (`GameInput` arrays in, `GameResult` + possession log out) so a 2D engine can be swapped in later.

---

## 1. Quantitative Genetics for Game Use

### 1.1 Additive (infinitesimal) model

- [SOURCED] Barton, Etheridge & Véber (Theor. Pop. Biol. 2017): offspring genetic value is normal around the midparent; within-family variance depends only on ancestral variance and the pedigree, not parental phenotypes.\[1\]
- [SOURCED] Offspring of diploid parents i, j: mean = midparent, segregation variance V₀·(1 − (F_i + F_j)/2) (Barton & Etheridge, CIRM slides).\[2\]
- [SOURCED] Under random mating, population variance converges to 2V₀ (offspring variance = V₁/2 + V₀), so half the variance is within families and half between (Etheridge, "Some mathematical models of evolution III").\[3\]
- [SOURCED] With dominance, inbreeding depression is a predictable mean shift proportional to identity-by-descent; within-family components stay Gaussian (Barton 2023, Genetics).\[4\]

**Game translation [INFERRED].** Work in phenotypic-SD units, founding phenotypic variance = 1.

- V_A = h²; V_E = 1 − h²; equilibrium V₀ = h²/2.
- Child A = (A_m + A_f)/2 + N(0, (h²/2)·(1 − (F_m + F_f)/2)).
- Phenotype z = A + E + ID + dev(age), with E ~ N(0, 1 − h²), ID = inbreeding shift, dev = development/aging.

**Regression to the mean is automatic.** Expected child phenotype = midparent *breeding value*; since E(A | z) = h²·z, the offspring-on-midparent slope is h². Parents 2 SD above the mean produce children ~1.6 SD above for height (h² 0.8) and ~0.8 SD above for touch (h² 0.4). Do not add an explicit regression step; it would double-count.

**Bulmer caveat [INFERRED].** Selection barely depletes V₀ in the pure infinitesimal model, so sustained selection gives unbounded response — the root of the "generational super-athletes" risk. Counters are in §4.

### 1.2 Heritability values

| Trait | Default h² | Basis |
|---|---|---|
| Height | 0.80 (0.70–0.90) | [SOURCED] Jelenkovic et al., eLife 2016, 40 twin cohorts with 143,390 complete twin pairs born 1886–1994 (~0.80); Jelenkovic et al., Sci Rep 2016, 45 cohorts (0.70–0.90 adolescence/adulthood); Silventoinen 2003 women ACE 0.68–0.84, AE 0.89–0.93 |
| Wingspan / ape index | 0.70 | [INFERRED] skeletal, like height |
| Hand size | 0.70 | [INFERRED] skeletal |
| Endurance (VO2max-like) | 0.50 (0.40–0.59) | [SOURCED] HERITAGE baseline VO2max maximal heritability 59% (maternal 36%), "at least 50%… inflated… by nongenetic factors" |
| Muscle fiber ratio → vertical/strength | 0.45–0.55 | [SOURCED] fiber distribution ≈45% (Simoneau & Bouchard 1995); vertical/strength [INFERRED] |\[5\]
| Lateral speed/agility | 0.50 | [INFERRED] |
| Learning rate | 0.47 | [SOURCED analog] VO2max trainability 47%, 28% maternal, with 2.5× more variance between families than within (Bouchard et al. 1999, J Appl Physiol) |
| Shooting touch | 0.40 | [INFERRED] mostly learned |
| Court vision | 0.45 | [INFERRED]; Joshi 2015 confirms directional dominance on cognition |
| Temperament | 0.40 | [INFERRED] |
| Injury resilience | 0.30 | [INFERRED] |
| Strength/mass | 0.55 | [INFERRED] |

No reaction-time heritability was retrieved; fold it into lateral speed and vision [INFERRED].

### 1.3 Inbreeding

**F from a pedigree [INFERRED: standard tabular coancestry method].** f(x,x) = (1 + F_x)/2; f(x,y) = (f(x, sire_y) + f(x, dam_y))/2 when y is not an ancestor of x; F_child = f(mother, father). Founders/commoners have f = 0 outside their pedigree. Checks: full-sib child 0.25, half-sib 0.125, first cousins 0.0625.

**Empirical magnitudes**

- [SOURCED] Height: first-cousin offspring 1.2 cm shorter (Joshi 2015, Nature; 354,224 individuals, 102 cohorts)\[6\] ≈ 19 cm per unit F ≈ 2.7 SD/F at SD ≈ 7 cm [INFERRED].
- [SOURCED] Cognition: first-cousin offspring have 10 months less education and "a decrease of 0.3 s.d. in g" (Joshi 2015). That is ≈4.8 SD/F on general cognition; keep the vision/learning default at 2.4 SD/F as a conservative half-strength value, with 4.8 as the sourced upper bound. Joshi found no effect on blood pressure, LDL or ten other cardio-metabolic traits.
- [SOURCED] Health: 3.5% excess pre-reproductive mortality (r² = 0.70, 69 study populations) and 1.1% excess infant mortality at F = 0.0625 (Bittles & Black 2010, PNAS), which the authors call "an upper-level estimate"; birth-defect excess 1.7–2.8%.
- [SOURCED] Fertility: no human deficit; first cousins had 0.08 *more* births, and "no major adverse associations with reproductive parameters such as miscarriages and fertility have been documented" (Geneva Workshop Report).\[7\]\[8\] A fertility penalty is a design choice.

### 1.4 Outbreeding, heterosis, and genetic distance

- [SOURCED] Frankham et al. 2011 (Conservation Biology): outbreeding-depression probability rises when populations are distinct species, have fixed chromosomal differences, exchanged no genes in 500 years, or occupy different environments; "populations in similar environments had not developed OD even after thousands of generations of isolation."\[9\]
- [INFERRED] Frankham's criteria give a numeric race distance: D = 0.35·karyotype + 0.25·isolation_time + 0.25·environment_divergence + 0.15·mean_trait_divergence (each 0–1).

| Pair | D [INFERRED] | Rationale |
|---|---|---|
| Human–Elf | 0.35 | Similar niches, long but incomplete isolation |
| Human–Goliath | 0.55 | Mountain/cold niche, major size divergence |
| Elf–Goliath | 0.75 | Maximal divergence on both axes |

**Effects of D [INFERRED; shapes follow the qualitative literature]:**

- Hybrid fertility: fert = exp(−2.0·D·H_anc) (H_anc = ancestry heterozygosity of the pair) → F1 Human×Goliath ≈ 0.33, Elf×Goliath ≈ 0.22.
- F1 heterosis: +0.15·D SD on endurance and injury resilience, halved each hybrid×hybrid generation.
- F2+ outbreeding depression: −0.2·D·(recombined ancestry heterozygosity) SD on the same traits.

### 1.5 Hybrid variance: where biology and the design goal diverge

[INFERRED; textbook Castle–Wright logic, not retrieved here.] For lines differing by Δμ across n effective loci, F1 is about as variable as the parents; F2 adds ≈ Δμ²/(8n). So *F2s and backcrosses are wider, not F1s.*

**Implementation.** Extra segregation variance per child: σ²_hyb = Σ_{r<s} 2·ā_r·ā_s·(μ_r − μ_s)²/(8·n_eff) × H_recomb, where ā = parental mean ancestry and H_recomb = 0 for pure×pure (F1), 1 for mixed×mixed. Default n_eff = 12 [INFERRED], raising Human×Goliath F2 height SD from ~8 to ~14 cm. **Design override (flagged):** to satisfy criterion (c) in F1 too, add fictional "developmental instability" noise N(0, (0.25·D)²) SD to all F1 traits, labeled as such in code.

### 1.6 Few-locus vs quantitative vs hybrid

| Model | Cost | Legibility | Recessives/inbreeding | Verdict |
|---|---|---|---|---|
| Few-locus (e.g., 10 loci × 12 traits) | 240 small ints/person | Poor | Natural | Too opaque |
| Pure quantitative | 12 floats | Good ("family is tall") | Mean shift only | No "carrier" stories |
| **Hybrid (recommended)** | 12 floats + ~6 loci + ancestry | Good | Rare conditions via real IBD | **Use** |

**Suggested loci [INFERRED, fictional], allele frequencies 0.02–0.10 per race:** *Titan* (recessive; +2 SD height, 3× injury hazard), *Hawkeye* (rare dominant; +1 SD vision), *Brittle* (recessive; −2 SD injury resilience), *Longwind* (additive; +0.5 SD endurance/copy), *Stonehands* (recessive; −1.5 SD touch), *Late-bloom* (recessive; learning window +3 years).

### 1.7 Crusader Kings 3 reference

- [SOURCED, community guides] Congenital traits have active and inactive (carrier) forms; both parents active gives about 80% inheritance (GameRant). Parents at the same level of a leveled trait have a 50% chance of a child one level higher (Paradox forum citing the wiki).\[10\]\[11\]\[12\]
- [SOURCED, community guide] Tier mismatch cuts inheritance ~20% per tier; Blood legacies give +30% good-trait inheritance, +30% reinforcement, −30% bad-trait inheritance (Steam "Eugenics 101"). That guide's alternative ladder (100/75/50/25%) conflicts with GameRant; treat exact numbers as unverified.\[13\]
- **Player confusion** [SOURCED, forums/guides; design-practice only]: carrier status is hidden ("There is no way to know if these traits were inherited or not," TheGamer); players misread which table applies; the Inbred chance is undiscoverable ("I could not find what this number is").\[10\]\[12\]\[13\]
- **Lessons [INFERRED]:** show expected child distributions (mean ± 1 SD) before a marriage, pedigree-inferred carrier probabilities, and numeric F on the family tree.

### 1.8 Build Spec: genetics

```python
@dataclass(slots=True)
class Genome:
    A: np.ndarray          # float32[N_TRAITS], additive breeding values, phenotypic-SD units
    loci: np.ndarray       # uint8[N_LOCI, 2], allele ids (0=wild, 1=variant)
    ancestry: np.ndarray   # float32[N_RACES], sums to 1
    F: float               # inbreeding coefficient
    E_perm: np.ndarray     # float32[N_TRAITS], permanent environment draw (fixed at birth)

def make_child(m, f, rng, P):
    Fm, Ff = m.genome.F, f.genome.F
    anc = 0.5 * (m.genome.ancestry + f.genome.ancestry)
    h2 = P.h2
    seg_var = (h2 / 2) * (1 - (Fm + Ff) / 2)
    H_rec = 1.0 if (is_mixed(m) and is_mixed(f)) else 0.0
    hyb_var = H_rec * pairwise_ancestry_term(anc, P.race_mu_sd_units, P.n_eff)
    D = ancestry_distance(m.genome.ancestry, f.genome.ancestry, P.D)
    f1_var = (0.25 * D) ** 2 if is_F1(m, f) else 0.0
    A = 0.5 * (m.genome.A + f.genome.A) + rng.normal(0, np.sqrt(seg_var + hyb_var + f1_var))
    loci = np.stack([m.genome.loci[np.arange(N_LOCI), rng.integers(0, 2, N_LOCI)],
                     f.genome.loci[np.arange(N_LOCI), rng.integers(0, 2, N_LOCI)]], axis=1)
    F = coancestry(m.id, f.id)                           # memoized tabular method
    E_perm = rng.normal(0, np.sqrt(1 - h2))
    return Genome(A, loci, anc, F, E_perm)

def adult_phenotype_z(g, P):
    z = g.A + g.E_perm - P.inbreeding_dep * g.F
    z += heterosis(g.ancestry, P) + mendelian_effects(g.loci, P)
    return z

def to_units(z, ancestry, P):
    mu = ancestry @ P.race_mu; sd = ancestry @ P.race_sd
    return mu + sd * z

def conception_prob(m, f, P):
    D = ancestry_distance(...); H = ancestry_heterozygosity(m, f)
    return P.base_fert[race_mix] * age_fert(m.age, race) * np.exp(-2.0 * D * H)
```

| Parameter | Value | Range | Basis |
|---|---|---|---|
| inbreeding_dep, height | 2.7 SD/F | 2–3.5 | [SOURCED] 1.2 cm at F=.0625 |
| inbreeding_dep, vision/learning | 2.4 SD/F | 1.5–3 | [INFERRED] from Joshi education effect |
| inbreeding_dep, injury resilience | 3.0 SD/F | 2–4 | [INFERRED] |
| Excess child mortality | +0.56 × F | 0.3–0.6 | [SOURCED] 3.5% at F=.0625 |\[14\]
| Fertility penalty from F | 0 (realistic) or −1.0×F (gameplay) | 0–2 | [SOURCED] no human deficit |\[7\]
| n_eff (hybrid) | 12 | 6–30 | [INFERRED] |
| Hybrid fertility k | 2.0 | 1–3 | [INFERRED] |

---

## 2. Basketball Performance Data and Validation Targets

### 2.1 League-level distributions

- [SOURCED] NBA 2023-24 (NBA.com Stats Survey): PPG 114.2, pace 99.2, OffRtg 114.5; FG% 47.4, 3P% 36.6, FT% 78.4, eFG% 54.7; 3PA/FGA 39.5%, FTA/FGA 0.244; TO% 13.6, OREB% 28.3.\[15\]
- [SOURCED] Basketball-Reference: 2019-20 pace 100.2, eFG% .528, TOV% 12.8, ORB% 22.6, FT/FGA .199, ORtg 110.4; 1984-85 pace 102.1, eFG% .496, TOV% 14.9, ORB% 32.9. ORB% fell steadily from ~33% (early 1980s) to 22.6%.\[16\]
- [SOURCED] Per-team per-game ranges (ESPN 2025-26 partial snapshots): FGA 83.5–93.4; 3PA 30–42; FTA 20.4–27.3; ORB 9.2–15.1; DRB 30–35; AST 24.5–30.1; STL 6.9–9.9; BLK 3.5–5.8; TOV 11.8–16.6; PF 18.2–21.8. Steals/turnovers ≈ 0.59 (8.6/14.6).
- [SOURCED] Ceiling reference: 2024-25 Thunder 68-14, MOV +12.87, ORtg 120.3, DRtg 107.6 (Basketball-Reference).\[17\]
- [SOURCED] Home court: ~3.24 points/game with ~1.3 more home FTA (Harris & Roebber 2019, PLOS One);\[18\]\[19\] 2.3 points in 2013–2018 (Cheng, per a 2024 systematic review);\[20\] historical 3.5 pts/100 and ~60% home wins, down to 53.7% and +2.2 in 2014-15 (ESPN/Haberstroh).\[21\]\[22\] Rest explains only ~0.3 points; a back-to-back costs ~1.77 points (Entine & Small).\[23\] Each +1 pp 3PA rate ≈ −0.2 pp home-win edge (Harvard Sports Analysis).\[24\]
- [SOURCED] Variance: per-game SD of 14.1 points/100 in home-minus-away margin (ESPN, since 1996).\[21\] Single-team score SD not found; use 11–13 [INFERRED].
- [SOURCED] Four Factors weights: Oliver 40/25/20/15; Poropudas 2023 re-derives 45/34/16/6.\[25\]\[26\]

**Conflict flagged:** NBA.com OREB% 28.3% vs Basketball-Reference ORB% 22.6% likely reflect different definitions.\[15\]\[16\] Define ORB% = ORB/(ORB + opp DRB) and target 22–28%.

### 2.2 Physical traits → performance

- [SOURCED] Cui et al. 2019 (Frontiers in Psychology; 3,610 Combine participants, 2000–2018): drafted players beat undrafted on height, wingspan, vertical, lane agility and ¾-court sprint (p < 0.01, ES 0.26–0.87); height and wingspan dominate all positions; leg power adds for guards, agility/speed for PF/C.\[27\]
- [SOURCED] Teramoto et al. 2018 (via Cui et al.): "length-size" items correlate r = 0.31–0.54 with Defensive Box Plus/Minus, less with offense.\[28\]\[29\]
- [SOURCED] Elite players: standing vertical > 60 cm, max vertical > 80 cm; forwards and guards sprint/change direction faster than centers (cited in Cui et al.).\[27\]
- [SOURCED] Dunk-contest finalists vs eliminated (PLOS One 2024): height 193.4 ± 9.5 vs 200.1 ± 6.5 cm; wingspan 207.3 ± 9.3 vs 213.2 ± 8.4 cm — a size-vs-explosiveness trade-off.\[30\]\[31\]
- [SOURCED] NBA centers average 2.11 m (Bullock et al. 2021).\[32\]
- Gap: no published regression links wingspan directly to per-possession block/steal probability; §3 coefficients are [INFERRED].

### 2.3 Aging and development

- [SOURCED] Functional data analysis, 645 NBA players (arXiv 1403.7548): peak 26.3; fastest gains 19–20 and 23–25; fastest declines 29–31 and 36–38.\[33\]
- [SOURCED] APBRmetrics regressed WS/48: 0.100 at 26, 0.085 at 30, 0.057 at 33, 0.022 at 36.\[34\]
- [SOURCED] Vaci et al. 2019 (PLOS One): post-peak decline is fast at first, then slows; development and decline are related.\[35\]
- [SOURCED] All-Star mean age 26.5, MVP 27.9 (Dartmouth Sports Analytics).\[36\]
- [SOURCED] Basketball GM ages athletic ratings from the late 20s while shooting/IQ modifiers "reverse most of the age-related decline": "You don't forget how to play the game as you get older, you mostly just lose your athleticism" (ZenGM blog 2014).\[37\]

### 2.4 Injuries

- [SOURCED] Bullock et al. 2021 (NBA 2008–2019; 302,018 player-games): 17.80 injuries+illnesses per 1000 game-exposures (15.60 injuries only); median 3 games missed (IQR 0–6); ankle 2.57, knee 2.44 per 1000; no position difference; incidence rises through the season; lockout season 21.68.\[38\]\[39\]
- [SOURCED] Mack et al. 2025: 13.8/1000 player-games, 6.0/10,000 player-minutes; 33.6–38.5% of injuries cost a game.\[40\]
- [SOURCED] Drakos et al. 2010: 19.1/1000, no correlation with age, height, weight or experience.\[41\]
- [SOURCED] Structural knee injuries per 1000 game exposures: <20 3.23; 20–24 2.26; 25–29 2.97; >30 3.11.\[42\]
- Implication [INFERRED]: keep incidence about flat with age; scale severity/recovery with age; drive risk by workload.

### 2.5 Build Spec: validation table

Targets for a Human-majority calibration league; race-heavy leagues should deviate.

| Metric | Target range | Basis |
|---|---|---|
| Pace | 96–102 | [SOURCED] 99.2–100.2 |\[15\]\[16\]
| ORtg | 108–116 | [SOURCED] 110.4–114.5 |\[15\]\[16\]
| PPG per team | 108–118 | [SOURCED] 114.2 |\[15\]
| eFG% | .510–.555 | [SOURCED] .528–.547 |\[15\]\[16\]
| TS% | .560–.590 | [SOURCED] team TS% .56–.62 (Lineups.com 2025-26) |
| TOV% | 12.0–14.5 | [SOURCED] 12.8–13.6 |\[15\]\[16\]
| ORB% | 22–29 | [SOURCED, conflicting defs] |\[15\]\[16\]
| FTA/FGA | .19–.26 | [SOURCED] .199–.244 |\[15\]\[16\]
| 3PA/FGA | .33–.42 | [SOURCED] .395 |\[15\]
| 3P% / FT% | .34–.38 / .75–.80 | [SOURCED] .366 / .784 |\[15\]
| AST / STL / BLK / PF per game | 23–30 / 7–10 / 3.5–6 / 18–22 | [SOURCED] ESPN |
| Home margin | +1.5 to +3.5 | [SOURCED] 2.2–3.24 |
| Home win % | 53–60% | [SOURCED] |
| Home–away margin SD | 12–15 pts/100 | [SOURCED] 14.1 |
| Team score SD | 10–14 | [INFERRED] |
| Best-team MOV | ≤ +13 | [SOURCED] OKC +12.87 |
| League win% SD (30 games) | 0.12–0.20 | [INFERRED]; luck floor 0.091 |
| Mean peak age (Human) | 25.5–28 | [SOURCED] 26.3 |\[33\]
| Injuries per 1000 player-games | 13–19 | [SOURCED] |\[38\]\[40\]
| Median games missed | 2–4 | [SOURCED] 3 |\[39\]

---

## 3. Basketball Simulation Engine Design

### 3.1 Survey

- [SOURCED] **Basketball GM (zengm, open-source TypeScript):**
  - A game is a loop of `simPossession()`: swap offense/defense, update composites, run inbound time (uniform 1–5 s after makes), resolve tov, stl, nonShootingFoul, drb, orb, fg, ft, timeout or out-of-bounds. Substitutions only on dead balls.\[43\]
  - Composites: blocking, fouling, passing, rebounding, stealing, turnovers, usage, jumpBall. Home advantage multiplies composites for home and divides for away (turnovers/fouling inverted).\[43\]
  - Constants: `fatigueFactor = 0.055`, energy recovery +0.016; `synergyFactor = 0.1` (playoffs: fatigue ÷1.85, synergy ×2.5); jump ball P = 0.5·(ratio)³; non-shooting foul 0.08 per check, halved in protect-the-lead spots, "calibrated to 2021-24 NBA play-by-play"; rushed shots `probMake *= sqrt(possessionLength/8)`; foul-trouble limits 2/3/5/5; `probAndOne = 0.05` (older commit).\[43\]\[44\]
  - Clock tactics: holdForLastShot, twoForOne (32–52 s), catchUp, maintainLead, intentional fouls; blowout bench logic (e.g., ≥20 points, < 7 min).\[43\]
  - Height rating "includes wingspan and standing reach" (manual); potential is only a 75th-percentile descriptive forecast.\[45\]
- [SOURCED] **Academic:** Cervone et al. (arXiv 1408.0777) model possessions as a multiresolution stochastic process with a logit shot model and turnovers as absorbing states — support for logit-link shot models.\[46\]
- [SOURCED] **Hobby text sims** (Trevor Johnson; Manav Gagvani): Monte Carlo loops on season shooting splits; stat-driven, not trait-driven.\[47\]\[48\]
- Commercial text-sim internals were not retrieved.

### 3.2 Recommended model: traits → composites → logits

**Principle [INFERRED].** Every probability is `sigmoid(logit(base) + Σ β_k · Δ_k)`, with Δ_k an offense-minus-defense difference in physical units or frozen-reference z-scores. Race never appears; styles emerge from shot-selection EV, contest geometry (reach vs release, quickness vs quickness) and rebounding mass/reach.

**Per-player derived quantities (per game, scaled by fatigue):**

- `reach = height·(1.33 + 0.5·(ape − 1.0))` cm [INFERRED]
- `max_reach = reach + vertical·fat^0.5`
- `quick = −z(lane_agility_s)·fat`
- `release = 0.92·reach + 0.4·vertical·fat`
- `handle = 0.45·z_vision + 0.35·quick + 0.20·z_hand`
- `reb = 0.45·z(max_reach) + 0.35·z_strength + 0.20·z_vertical`
- `usage = softmax_weight(0.6·z_touch + 0.4·z_vision + 0.3·z_temperament + coach_bias)`

```text
simulate_possession(off, def, state, rng):
    t = sample_gamma(mean = 14.4 s * pace_adj(off.quick_mean, def.quick_mean, coach))
    # 1. Non-shooting foul
    if rng < p_nsf = 0.045 * exp(0.25*(def.reach_clumsiness - 0))  -> FT if bonus else new poss
    # 2. Turnover
    Δ_tov = mean(def.quick + 0.5*z(def.wingspan)) - off.ballhandler.handle
    p_tov = sigmoid(logit(0.130) + 0.35*Δ_tov - 0.10*z(off.ballhandler.temperament))
    if rng < p_tov: steal with prob 0.59*sigmoid(0.4*Δ_tov) (credit defender ~ quick+wingspan); end
    # 3. Shooter (usage softmax), shot type (EV softmax)
    shooter = choice(off5, w = usage)
    for type in {rim, mid, three}:
        d = assigned defender (position-sorted by height) or rim help
        c_type = contest(type, shooter, d)
        p_make_est[type] = make_prob(type, shooter, c_type)
    EV[type] = pts[type]*p_make_est[type]*(1 - p_block_est) + foul_value[type]
    type = choice(softmax((EV + coach.style_bias[type]) / τ = 0.12))
    # 4. Block (rim & mid)
    p_blk = sigmoid(logit(base_blk[type]) + 0.9*(d.max_reach - shooter.release)/10cm)
    # 5. Shooting foul
    p_sf = sigmoid(logit(base_sf[type]) + 0.3*(z(shooter.strength)-z(d.strength)) - 0.3*d.quick_rel)
    # 6. Make
    p_make = sigmoid(logit(base_make[type]) + β_touch[type]*z(shooter.touch_eff)
                     - β_c[type]*c_type + home_shift + clutch_shift)
    resolve: foul+make -> and-one; foul+miss -> 2/3 FT; make -> pts; miss -> rebound
    # 7. Rebound
    Δ_reb = mean(off.reb) - mean(def.reb)
    p_orb = sigmoid(logit(0.255) + 0.55*Δ_reb - 0.25*[type==three])
    rebounder = choice(team, w = exp(1.2*reb_i))
    # 8. Energy, assists (p_ast by type * passer vision), stats, log
```

**contest(type, s, d):**

| Type | Formula |
|---|---|
| rim | `sigmoid(0.12*(d.max_reach - s.release) + 0.8*(d.quick - s.quick))` |
| mid | `sigmoid(0.08*(d.max_reach - s.release) + 1.0*(d.quick - s.quick))` |
| three | `sigmoid(0.04*(d.max_reach - s.release) + 1.3*(d.quick - s.quick) - 0.5)` |

**Default coefficients [INFERRED starting points; calibrate per §3.5]:**

| Parameter | rim | mid | three | FT |
|---|---|---|---|---|
| base_make | 0.64 | 0.42 | 0.365 [SOURCED 3P%] | 0.784 [SOURCED] |\[15\]
| β_touch | 0.25 | 0.45 | 0.55 | 0.60 |
| β_c | 1.10 | 0.80 | 0.65 | — |
| base_blk | 0.09 | 0.03 | 0.01 | — |
| base_sf | 0.17 | 0.05 | 0.015 | — |
| β_size_at_rim (s.release − 300 cm)/10 | +0.20 | 0 | 0 | — |

| Parameter | Value | Basis |
|---|---|---|
| τ | 0.12 | [INFERRED] |
| p_tov base | 0.130 | [SOURCED] TOV% 12.8–13.6 |\[15\]\[16\]
| p_orb base | 0.255 | [SOURCED] 22.6–28.3% midpoint |\[15\]\[16\]
| Steal share of TOV | 0.59 | [SOURCED] ESPN |
| Home make shift | +0.025 logit; home shooting-foul ×1.06 → ≈ +1.3 FTA | [INFERRED]; FTA target [SOURCED] |
| Energy drain/possession | 0.0045·(1 − 0.15·z_endurance)·(mass/100 kg)^0.5 | [INFERRED] |
| Bench recovery/possession | 0.012 | [INFERRED] |
| fat | 1 − 0.055·(1 − energy)/0.1, floor 0.7 | [SOURCED form: BBGM 0.055; scaling INFERRED] |\[43\]
| Clutch shift (last 5 min, ≤5 pts) | +0.10·z_temperament | [INFERRED] |

**Coach/substitution hooks:** rotation by `ovr_game × fat`; sub out below energy 0.72; foul limits 2/3/5/5 [SOURCED BBGM]; coach parameters `style_bias[type]`, `pace_pref`, `rotation_depth`, `foul_aggression`.\[43\]

**Why styles emerge [INFERRED]:** Goliaths' max_reach sits 60–80 cm above Humans, so rim contests against them are weak and their blocks frequent — but slow agility loses closeouts (opponent three EV rises), mass drains energy, and low touch hurts FT% and threes. Elves' touch and quickness raise three EV and cut turnovers, while low strength/reach loses rebounds and rim fouls. High Human variance makes Human rosters differ from each other.

### 3.3 Speed and structure

- Workload: 120 games/season; 50 years ≈ 6,000 games ≈ 1.2M possessions (100 years ≈ 2.4M).
- [INFERRED] Pure Python at ~20–40 µs/possession gives 24–48 s for 50 years — too close to 60 s. A numba `@njit` kernel on flat float32 arrays should reach ~0.5–2 µs/possession.
- [SOURCED] Loose reference: ZenGM's JavaScript basketball benchmark fell from 12.78 to 10.54 s after allocation fixes (PR #543);\[49\] benchmark size not stated.
- Layout: `engine/kernel.py` holds one `@njit(cache=True)` `sim_game(off_arrays, def_arrays, params, seed) -> (box[2,P,NSTAT], log[n_poss, 8])` with no Python objects; `prange` over each day's 4 games; `np.random.seed(seed)` per game inside numba, with seeds derived outside from `SeedSequence`; optional compact log [period, clock, off_team, event_code, player_a, player_b, pts, flags].
- Swappability: `class GameEngine(Protocol): def simulate(self, gi: GameInput, seed: int) -> GameResult`. GameInput carries only player physical/skill arrays, lineups, coach parameters and home flag.

### 3.4 Build Spec: engine formulas

All formulas as in §3.2; constants in `config/engine.yaml`, every β named, every base rate traced to a validation row.

### 3.5 Calibration procedure

1. Build 8 Human-only teams from the "draft-selected" distribution (top ~2% on a composite, height ≈ 198 ± 8 cm) [INFERRED].
2. Simulate 20 seasons × 120 games at fixed seeds.
3. Loss L = Σ ((sim − target_mid)/half_width)² over the validation table.
4. Optimize base rates first (base_make ×4, p_tov, p_orb, base_sf, pace) with Nelder–Mead or CMA-ES (≈200 evaluations × 2,400 games).
5. Then tune slopes (β_c, β_touch, reach coefficients) on blocks, steals, MOV spread, best-team MOV ≤ 13.
6. Freeze parameters with a config hash; mixed-race leagues are for style diagnostics only, never re-tuning.
7. CI asserts the Human league stays inside the table.

---

## 4. Dynasty, Management and Emergent-Story Systems

### 4.1 How reference games keep leagues stable

- [SOURCED] **Basketball GM** (source code and developer blog, via subagent): each season every non-height rating changes by `bound((baseChange + ageModifier) × uniform(0.4, 1.4), limits)`; base change +2 (≤21), +1 (22–25), 0 (26–27), −1 (28–29), −2 (30–31), −3 (32–34), −4 (35–40); noise σ = 5, bounded [−4, 20] at ≤23; speed −2/yr at 28–30 and −3 at 31–35; jumping −3 at 27–30;\[50\] coaching ±25% on progs.\[51\] Draft prospects use a hardcoded age-mix formula, and new leagues are seeded from "actual simulated past draft classes."\[52\]\[53\] The developer's 2023 test (ZenGM blog, Aug 14, 2023) found old and new code identical in freshly created leagues (70+ OVR players 8.5 vs 8.6 over 1,000 runs), but after 20 simmed seasons a coaching-spending bug produced "significant differences": 70+ OVR players 7.9→8.9, 80+ 0.3→0.5, top-team OVR 75.5→82.5. A one-time 2018 rescale moved old 100s to "more like a 75 or 80."\[52\]
- [SOURCED] **Out of the Park Baseball** (official wiki/manuals): aging/development speed modifiers (default 1.000; "0.500 should roughly halve the expected results"); Talent Change Randomness 1–200 (default 100); "ratings relative to leagues" is display-only;\[54\]\[55\] league-total modifiers with auto-adjust keep *statistics* on target (forum-sourced).\[56\]\[57\]
- [SOURCED] **Football Manager** (official site, community-written): newgen quality comes from hidden nation Youth Rating and Game Importance plus club Youth Facilities, Junior Coaching and Youth Recruitment.\[58\] Community tests dispute the facilities effect and attribute ~40% of potential to Junior Coaching (FM-Arena, unofficial).\[59\]\[60\] FM24 youth ratings (sortitoutsi, unofficial): Brazil 163, Germany/France 155, England 135.\[61\]
- **CK3 succession** mechanics were not retrieved; the PoC uses eldest-child succession.

**Pattern [INFERRED]:** stable sims pin *inflow* talent to a stationary distribution and age it with fixed curves. Breeding makes inflow endogenous, which is why the commoner anchor is mandatory.

### 4.2 Failure modes → counter-mechanisms

| Failure mode | Cause in this design | Counter-mechanism (default) |
|---|---|---|
| Generational super-athletes | V₀ barely depletes; assortative dynasty marriages | (1) Trade-offs: injury hazard × exp(0.5·max(0, z_height − 2)); lateral speed −0.3 SD per +1 SD height beyond +1.5; energy drain ∝ √mass. (2) Viability selection: child mortality +2%·max(0, |z| − 3)² on height/strength. (3) Mean reversion: `A = 0.95·A + …` toward race mean, segregation rescaled to keep V_A [INFERRED]. (4) Inbreeding depression punishes closed lines. |
| League talent drift | Endogenous breeding | Commoner pool: 3–4 new players per team per year from fixed race distributions, selected on a fixed composite; commoners marry into dynasties |
| Hidden inflation | z-scores against current league | Frozen year-0 reference for all z-scores and dashboards |
| Runaway dominant team | Family players auto-join | Cap 4 house members on house team; others enter the draft, with one right of first refusal costing the family's first-round pick; reverse-standings draft; roster cap 13 |
| Population collapse | Low Elf fertility, hybrid infertility | Per-race target population with density-dependent fertility; commoner immigration |
| Population explosion | High fertility | fert × (1 − N/K_race) |
| Dynasty extinction | No heir | Sibling → nephew → closest kin by coancestry → cadet branch adopts; log as story event |
| Elves dominate forever | Long peak | Lower Elf strength/reach ceiling; low fertility; ~45-year generation interval [INFERRED] |

### 4.3 Emergent history generation

- **Records:** running game/season/career maxima and minima per stat, per race and overall; emit an event on each break.
- **Awards:** MVP = top season value, where season value = Σ possessions × points added vs replacement (a simple box plus-minus with Four-Factor weights [SOURCED 40/25/20/15]).\[26\]
- **Hall of Fame [INFERRED]:** career value threshold calibrated to ~1.0–1.5 inductees/season, plus a "dynasty contribution" wing for heads of house.
- **Rivalry index [INFERRED]:** +1 per game, +2 per game decided by ≤3, +5 per playoff series, +8 per marriage/inheritance dispute, +4 per player moving between the houses; decay ×0.85/season; top-3 rivalries get narrative hooks.
- **Triggers:** metrics beyond the 99.5th historical percentile; upsets at pregame win probability < 15%; heir death before 30; F1 hybrid prodigy (top-5 season value before 23); exposed recessive homozygotes; title droughts > 20 years ended.
- **Retellability:** each event stores (year, actors, houses, metric, percentile, prior context ids); the renderer chains events sharing actors into "arcs" that test (d) counts.

### 4.4 Build Spec: stability checklist

1. Year-0 reference population frozen and serialized.
2. Commoner pool regenerated yearly from fixed race parameters, supplying ≥40% of roster slots.
3. House-member cap enforced.
4. Trade-off matrix and viability selection active.
5. Mean-reversion term active.
6. Density-dependent fertility per race.
7. Per-decade dashboards: mean and 99th percentile of each trait (league and population), mean F, top-team MOV, win% SD, title Gini.
8. Alarms: any trait's league mean drifting > 0.5 reference SD per 50 years; top MOV > 15; one house > 40% of titles per 30 years.

---

## 5. Feeder/Vassal Systems (architecture notes only)

Not researched in this session; statements are [INFERRED/general knowledge, unverified]:

- **NBA G League:** mostly one-to-one affiliates; the parent holds player rights (two-way contracts) and funds operations.
- **MLB:** multi-level affiliates under Player Development License agreements; the parent pays salaries.
- **European football:** loans with wage splits and recall/option-to-buy clauses; promotion/relegation reshapes tiers yearly.

**PoC fields to add now (logic later):** `Team.parent_team_id: Optional[int]`, `Team.tier: int`; `Contract{player_id, rights_holder_team_id, employer_team_id, start, end, salary, loan_terms: Optional[dict]}` (rights separate from employment); `House.vassal_of: Optional[house_id]`, `House.obligations: list`; League = list of `Division{tier}` with `promotion_rules` defaulting to none.

---

## 6. Fantasy-Race Balance

### 6.1 Principles

- [INFERRED] Each race gets at least one physical *and* one cognitive/cultural strength, and a weakness of each kind.
- [INFERRED] Ground differences in fictional physiology and scaling laws: square-cube scaling (mass ∝ height³, muscle cross-section ∝ height², so relative strength and jump height don't scale up with size; not retrieved here), maturation timing, lifespan, and culture entered as an *environmental* component.

### 6.2 Avoiding real-world stereotype mapping [INFERRED design practice]

- No race gets lower *general* intelligence. Goliath "slow to learn" is slower *fine-motor learning* paired with a cognitive strength (composure, court memory).
- Keep between-race gaps on vision and temperament under 1 within-race SD so individuals routinely cross over.
- Put culture in the environment: house/region training-tradition vectors (e.g., Elf "apprenticeship" boosts touch in the window) that also benefit a Human raised there.
- Avoid real-world ethnic coding in names, art notes, dialect or geography; justify every difference physiologically.

### 6.3 Build Spec: race defaults

Trait means/SDs are in the Consolidated Parameter Table (adult, sex-averaged, pre-draft-selection; Human anchors [INFERRED]; Humans get 1.15× SDs for "high-variance baseline"). Justifications [INFERRED]: Goliath height ≈ +7 Human SD with long levers (ape 1.06); square-cube scaling lowers Goliath vertical, agility and endurance while raising absolute strength and hand size; Elves are slightly taller, slender and light (higher vertical, quickness, endurance; lower strength); Elf touch and vision edges stay < 1 SD; Goliath learning rate 0.75× reflects fine-motor learning; composure is the Goliath cognitive strength; injury resilience is lower for Goliaths (joint load) and Elves (light frame).

**Life history [INFERRED]:**

| Parameter | Human | Goliath | Elf |
|---|---|---|---|
| Growth complete | 20 | 24 | 30 |
| Critical learning window (learning ×) | 12–18 (×2.0) | 14–22 (×1.6) | 22–36 (×2.2) |
| Athletic peak | 25–28 | 26–29 | 32–48 |
| Athletic decline multiplier | 1.0 | 1.3 | 0.4 |
| Lifespan | 72 ± 12 | 60 ± 10 | 160 ± 25 |
| Fertile ages (female) | 18–42 | 20–40 | 40–110 |
| Base annual conception prob | 0.25 | 0.20 | 0.06 |
| Mendelian allele example | Titan 0.03 | Titan 0.10 | Hawkeye 0.08 |

**Aging implementation:**

- age_eff = (age − growth_offset_race)/timescale_race + 20.
- Apply ZenGM-shaped athletic decline on age_eff (speed −2/yr at 28–30, −3 at 31–35; vertical −3 at 27–30; endurance −2 at 31–35) [SOURCED shape], at 0.1 SD per ZenGM point [INFERRED].
- Skill growth: Δskill = lr · window_mult(age) · k · (ceiling − skill) + N(0, 0.08) SD, k = 0.18, ceiling = genetic z + 1.5 [INFERRED]; skills decline only 0.02 SD/yr after 33 (Human-equivalent).

---

## 7. Implementation Plan

### 7.1 Repo structure

```text
dynasty_sim/
  config/        defaults.yaml, races.yaml, engine.yaml (frozen calibrated params + hash)
  core/          rng.py (SeedSequence tree), ids.py, types.py, reference.py (frozen year-0 stats)
  genetics/      genome.py, inheritance.py, pedigree.py (coancestry memo), mendelian.py, hybrid.py
  development/   growth.py, learning.py, aging.py, injury.py
  population/    demography.py, succession.py, commoners.py
  league/        schedule.py, season.py, draft.py, roster.py, standings.py, playoffs.py
  engine/        interface.py, composites.py, kernel.py (numba), calibrate.py
  history/       records.py, awards.py, hof.py, rivalry.py, events.py, arcs.py
  output/        boxscore_md.py, career_md.py, family_tree.py (Markdown + DOT), export.py
  cli.py         run --years 100 --seed 42 --out runs/
tests/
```

### 7.2 Schemas (core fields)

- **Person:** id, name, house_id, sex, race_ancestry[3], birth_year, death_year, mother_id, father_id, spouse_ids, genome, phenotype_now[12], skills_now[4], energy_base, injury{type, games_left}, team_id, contract_id, career_stats_ref, is_head.
- **House:** id, name, head_id, heir_policy="eldest_child", team_id, tradition[4], prestige, vassal_of=None.
- **Team:** id, house_id, name, parent_team_id=None, tier=0, roster[≤13], coach{style_bias[3], pace_pref, rotation_depth, foul_aggression}.
- **Season:** year, schedule, results, standings, awards, champion. **Game:** id, season, day, home, away, score[2], box, seed, log_ref.
- **PossessionLog** (structured array): game_id, period, clock, off, event, pa, pb, pts, flags.
- **Storage:** Parquet (logs/stats), JSON (config, run manifest), Markdown (readable pages).

### 7.3 Libraries and seeds

- numpy, numba, pyarrow/pandas (output only), pyyaml, pytest, hypothesis, optional graphviz.
- `root = SeedSequence(run_seed)`; `root.spawn(5)` → [genetics, demography, league, engine, narrative]; per-season children by year; per-game seed = hash(engine_stream, season, game_id); all seeds in the run manifest.

### 7.4 Milestones with tests

| # | Milestone | Test |
|---|---|---|
| M0 | Skeleton, config, RNG tree | Same seed → byte-identical 1-year dummy output |
| M1 | Genetics + pedigree F | Full-sib child F = 0.25, first-cousin 0.0625; offspring-on-midparent slope within ±0.05 of h² over 20k families; variance stable over 50 random-mating generations |
| M2 | Demography + succession | Population within ±25% of K per race at 100 years; ≥6 of 8 houses survive in ≥90% of 50 seeds |
| M3 | Composites + numba single game | Box sums consistent; no NaN; 1,000 games < 1 s |
| M4 | Calibration | All §2.5 metrics in range for a Human league over 20 seasons |
| M5 | Season loop, draft, aging, injuries | Human mean peak age 25.5–28; injuries 13–19/1000 |
| M6 | Dynasty integration | Dashboards within alarms for 100 years across 20 seeds |
| M7 | History and outputs | Test (d) |
| M8 | Performance and acceptance | Tests (a)–(e) |

### 7.5 Acceptance test suite

**(a) Distinct styles without race bonuses.** Static: `GameInput` has no race/ancestry field; grep-guard no `race` token in `engine/`. Dynamic: over 30 seasons, regress team style vectors [3PA rate, rim share, ORB%, TOV%, pace, BLK/game] on roster mean ancestry; pass if R² ≥ 0.30 on ≥3 of 6 components, with Goliath share → +rim, +ORB%, +BLK and Elf share → +3PA, −TOV%. Control: permuting traits while holding ancestry labels drops R² below 0.05.

**(b) No inflation or collapse over 100 years.** Versus the frozen reference: each trait's league mean at year 100 within ±0.5 SD of the years 10–20 mean, 99th percentile within ±0.75 SD; Human-majority seasons inside §2.5 in ≥90% of seasons; each race > 0.5·K; ≥5 houses alive; 10 seeds.

**(c) Hybrid variance.** 5,000 offspring each of Human×Human, Goliath×Goliath, Human×Goliath F1 and F2. Pass if F1 SD > both pure lines on ≥8 of 12 traits (F1 override) and F2 > F1 on height and strength; Levene p < 0.01.

**(d) Retellable 50-year history.** ≥10 arcs of ≥3 linked events; ≥3 rivalries above threshold; ≥1 record per decade; ≥1 succession with an on-court consequence (> 0.15 win% change within 3 seasons); Markdown renders with no unresolved ids.

**(e) Performance.** `python -m dynasty_sim.cli run --years 50 --teams 8 --seed 1 --no-possession-log` < 60 s on a 4-core laptop, excluding cached numba compile; 100 years < 120 s.

---

## Consolidated Parameter Table

| Trait | h² | Human μ (σ) | Goliath μ (σ) | Elf μ (σ) | Inbreeding δ (SD/F) | Primary engine use |
|---|---|---|---|---|---|---|
| Height (cm) | 0.80 [S] | 176 (8.0) | 232 (11) | 189 (6) | 2.7 [S-derived] | reach, release, contest, rebound |
| Ape index | 0.70 [I] | 1.00 (.035) | 1.06 (.03) | 1.02 (.025) | 1.0 [I] | reach, steals |
| Vertical (cm) | 0.55 [I] | 50 (10) | 38 (7) | 56 (7) | 1.5 [I] | max_reach, rebound |
| Lane agility (s) | 0.50 [I] | 12.0 (.65) | 13.5 (.6) | 11.5 (.45) | 1.5 [I] | quick: contest, TOV, closeouts |
| Endurance | 0.50 [S] | 50 (11) | 40 (8) | 58 (8) | 1.5 [I] | energy drain |
| Strength | 0.55 [I] | 50 (13) | 82 (10) | 36 (8) | 1.5 [I] | rebound, fouls drawn |
| Hand length (cm) | 0.70 [I] | 20.5 (1.3) | 27 (1.5) | 21 (1.0) | 1.0 [I] | handle |
| Touch | 0.40 [I] | 50 (14) | 36 (9) | 62 (9) | 1.5 [I] | make prob, FT |
| Vision | 0.45 [I] | 50 (14) | 48 (11) | 58 (10) | 2.4 [S-derived] | handle, assists, usage |
| Learning rate | 0.47 [S-analog] | 1.00 (.23) | 0.75 (.12) | 1.05 (.15) | 2.4 [I] | skill growth |
| Injury resilience | 0.30 [I] | 50 (14) | 42 (10) | 44 (9) | 3.0 [I] | injury hazard |
| Temperament | 0.40 [I] | 50 (17) | 62 (10) | 52 (10) | 1.0 [I] | clutch, TOV, fouls |

**Engine:** base_make (rim/mid/3/FT) .64/.42/.365/.784; β_touch .25/.45/.55/.60; β_contest 1.10/.80/.65; base_blk .09/.03/.01; base_sf .17/.05/.015; p_tov .130; p_orb .255; steal share .59; τ .12; home make shift +.025; home foul ×1.06; fatigue factor .055.

**Genetics:** segregation variance (h²/2)(1 − F̄_parents); n_eff 12; D (H–E/H–G/E–G) .35/.55/.75; hybrid fertility exp(−2·D·H); F1 instability SD .25·D; child mortality +.56·F.

[S] = sourced, [I] = inferred.

---

## Known Unknowns and Risks

1. **ORB% definitions conflict** (28.3% vs 22.6%); target is wide (22–29%) — pick one definition and document it.\[15\]\[16\]
2. **No per-possession trait→outcome regressions** were found; all engine slopes are inferred. Teramoto's r = 0.31–0.54 with DBPM validates direction only.\[29\]
3. **Hybrid F1 variance:** biology says F1s are not wider; criterion (c) at F1 relies on a labeled fictional override.
4. **No human fertility penalty from inbreeding;** any such cost is gameplay.\[7\]
5. **The 3.5% excess mortality is an upper-level estimate** confounded by socioeconomic status.\[62\]
6. **Heritabilities for touch, vision, temperament, agility, injury resilience and reaction time** are inferred.
7. **CK3 percentages come from conflicting community guides.**\[11\]\[13\]
8. **FM facilities effect is disputed;** OOTP league-total auto-calibration is documented mainly in forums.
9. **Selection response is effectively unbounded** under the infinitesimal model; stability rests on inferred counter-mechanisms, with test (b) as the only guard. Expect tuning.
10. **Elf life history** yields only 1–2 Elf generations in 100 years; watch Elf house survival.
11. **Performance estimates are inferred;** profile at M3.
12. **Feeder-system facts (§5)** are unverified placeholders.
13. **Team score SD, league win% SD and the draft-selection rule** (≈198 ± 8 cm) are inferred calibration choices.

## Sources

1. [Theoretical Population Biology The 2020 Feldman Prize](https://rosenberglab.stanford.edu/papers/Rosenberg2020-TPB.pdf)
2. [the infinitesimal model - CIRM](https://www.cirm-math.fr/ProgWeebly/Renc1774/BartonEtheridge.pdf)
3. [Some mathematical models of evolution III: The inﬁnitesimal model](https://cancerdynamics.columbia.edu/sites/cancerdynamics.columbia.edu/files/content/Some%20mathematical%20models%20of%20evolution%20-Alison%20Etheridge,%20PhD%20The%20infinitesimal%20model%20.pdf)
4. [The infinitesimal model with dominance Nicholas H. Barton](https://research-explorer.ista.ac.at/download/14452/14469/2023_Genetics_Barton.pdf)
5. [www.frontiersin.org](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2019.00262/epub)
6. [Directional dominance on stature and cognition in diverse human populations - Experts@Minnesota](https://experts.umn.edu/en/publications/directional-dominance-on-stature-and-cognition-in-diverse-human-p/)
7. [Consanguineous marriages, pearls and perils: Geneva International Consanguinity Workshop Report - Genetics in Medicine](<https://www.gimjournal.org/article/S1098-3600(21)03648-0/fulltext>)
8. [Consanguineous marriages, pearls and perils: Geneva International Consanguinity Workshop Report - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1098360021036480)
9. [Predicting the probability of outbreeding depression - The Australian Museum](https://publications.australian.museum/predicting-the-probability-of-outbreeding-depression/)
10. [Crusader Kings 3: Genetic Traits Guide](https://www.thegamer.com/crusader-kings-3-genetic-traits-guide/)
11. [Crusader Kings 3: The Best Congenital Traits](https://gamerant.com/crusader-kings-3-best-congenital-traits/)
12. [Help with congenital traits](https://forum.paradoxplaza.com/forum/threads/help-with-congenital-traits.1442296/)
13. [Steam Community :: Guide :: Eugenics 101: Perfecting Offspring Through Incest](https://steamcommunity.com/sharedfiles/filedetails/?id=2222228440)
14. [Evolution in health and medicine Sackler colloquium: Consanguinity, human evolution, and complex diseases - PubMed](https://pubmed.ncbi.nlm.nih.gov/19805052/)
15. [2023-24 NBA Stats Survey: League scoring averages](https://www.nba.com/news/2023-24-nba-stats-survey-league-scoring-averages)
16. [NBA Advanced League Averages - Per Game](https://www.basketball-reference.com/tools/share.fcgi?id=B9Trg)
17. [2024-25 NBA Team Ratings](https://www.basketball-reference.com/leagues/NBA_2025_ratings.html)
18. [NBA team home advantage: Identifying key factors using an artificial neural network - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6668839/)
19. [The Subtle Biases That Influence Home-Court Advantage - AIP.ORG](https://www.aip.org/inside-science/the-subtle-biases-that-influence-home-court-advantage)
20. [The Influence of Home-Court Advantage in Elite Basketball: A Systematic Review - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11503446/)
21. [NBA - Analyzing diminishing value of home-court advantage - ESPN](https://insider.espn.com/nba/story/_/id/12243076/nba-analyzing-diminishing-value-home-court-advantage)
22. [Haberstroh: Home-court disadvantage in NBA](https://www.espn.com/nba/story/_/id/12241619/home-court-advantage-decline)
23. [The Role of Rest in the NBA Home-Court Advantage Oliver Entine](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf)
24. [NBA Home-Court Advantage is in Decline, Are 3s to Blame?](https://harvardsportsanalysis.org/2017/03/nba-home-court-advantage-is-in-decline-are-3s-to-blame/)
25. [Dean Oliver’s Four Factors Revisited](https://arxiv.org/html/2305.13032v1)
26. [Four Factors](https://www.basketball-reference.com/about/factors.html)
27. [Key Anthropometric and Physical Determinants for Different Playing Positions During National Basketball Association Draft Combine Test - PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6820507/)
28. [(PDF) Key Anthropometric and Physical Determinants for Different Playing Positions During National Basketball Association Draft Combine Test](https://www.researchgate.net/publication/336718597_Key_Anthropometric_and_Physical_Determinants_for_Different_Playing_Positions_During_National_Basketball_Association_Draft_Combine_Test)
29. [The NBA Combine, Correlation, and Tryouts: Individuality Matters! — Volt Performance Blog](https://blog.voltathletics.com/home/2018/5/14/the-nba-combine-correlation-and-tryouts)
30. [Anthropometric and physical fitness indicators in the ...](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0299262&type=printable)
31. [(PDF) Anthropometric and physical fitness indicators in the combine draft between the finalist and the eliminated player in the national basketball association all-star slam dunk contest](https://www.researchgate.net/publication/378040264_Anthropometric_and_physical_fitness_indicators_in_the_combine_draft_between_the_finalist_and_the_eliminated_player_in_the_national_basketball_association_all-star_slam_dunk_contest)
32. [Original Research Temporal Trends and Severity in Injury](https://journals.sagepub.com/doi/pdf/10.1177/23259671211004094)
33. [Functional Data Analysis of Aging Curves in Sports](https://arxiv.org/pdf/1403.7548)
34. [Aging curves - APBRmetrics](https://www.apbr.org/metrics/viewtopic.php?t=125)
35. [Large data and Bayesian modeling—aging curves of NBA players - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690864/)
36. [Peak Age in Sports](https://sites.dartmouth.edu/sportsanalytics/2021/11/10/peak-age-in-sports/)
37. [Revamped player development algorithm « Blog « ZenGM](https://zengm.com/blog/2014/09/revamped-player-development-algorithm/)
38. [Temporal Trends and Severity in Injury and Illness Incidence in the National Basketball Association Over 11 Seasons - Garrett S. Bullock, Tyler Ferguson, Jake Vaughan, Desiree Gillespie, Gary Collins, Stefan Kluzek, 2021](https://doi.org/10.1177/23259671211004094)
39. [Temporal Trends and Severity in Injury and Illness Incidence in the National Basketball Association Over 11 Seasons - PubMed](https://pubmed.ncbi.nlm.nih.gov/34179200/)
40. [Epidemiology of Injuries Among National Basketball Association Players: 2013-2014 Through 2018-2019 - Christina D. Mack, Mackenzie M. Herzog, Travis G. Maak, Asheesh Bedi, Rahul Gondalia, Peter Meisel, Frederick M. Azar, Jimmie Mancell, Aaron Nelson, John DiFiori, 2025](https://journals.sagepub.com/doi/10.1177/19417381241258482)
41. [Injury in the National Basketball Association](https://www.markdrakosmd.com/pdfs/injury-in-the-national-basketball-association-a-17-year-overview-july-august-2010.pdf)
42. [Knee Injuries and Associated Risk Factors in National Basketball Association Athletes - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666061X2200102X)
43. [zengm/src/worker/core/GameSim.basketball/index.ts at master · zengm-games/zengm](https://github.com/zengm-games/zengm/blob/master/src/worker/core/GameSim.basketball/index.ts)
44. [More tweaking - ratings are balanced now · zengm-games/zengm@0f7f68c](https://github.com/zengm-games/zengm/commit/0f7f68c72f9f642e81391fdba91a990c0eb5bcdd)
45. <https://basketball-gm.com/manual/>
46. [A Multiresolution Stochastic Process Model for Predicting Basketball Possession Outcomes](https://arxiv.org/pdf/1408.0777)
47. [NBA Game Simulation - Trevor Johnson](https://trevor-johnson.github.io/NBA_Simulation/)
48. [How I built a complete NBA game simulator with less than 500 lines of code - Manav's Musings](https://manav.gagvani.com/index.php/2021/08/04/how-i-built-a-complete-nba-game-simulator-with-less-than-500-lines-of-code/)
49. [Reduce simulation allocations and repeated work across all four sports by sync0516 · Pull Request #543 · zengm-games/zengm](https://github.com/zengm-games/zengm/pull/543)
50. [zengm/src/worker/core/player/developSeason.basketball.ts at master · zengm-games/zengm](https://github.com/zengm-games/zengm/blob/master/src/worker/core/player/developSeason.basketball.ts)
51. [The finances revamp made coaching and health spending too powerful « Blog « ZenGM](https://zengm.com/blog/2023/08/finances-revamp-bug/)
52. [Player ratings and development beta! « Blog « ZenGM](https://zengm.com/blog/2018/02/player-ratings-and-development-beta/)
53. [Two new league settings: "Age of Draft Prospects" and "Force Retire at Age" « Blog « ZenGM](https://zengm.com/blog/2021/03/age-draft-prospects-force-retire-age/)
54. [Out of the Park Developments Online Manuals](https://manuals.ootpdevelopments.com/index.php?man=ootp17&page=game_setup_page.options_player)
55. [Game Settings - OOTP Wiki](https://wiki.ootpdevelopments.com/index.php?title=OOTP_Baseball%3AScreens_and_Menus%2FGame_Menu%2FGame_Settings)
56. [Auto-Calc Modifiers - OOTP Developments Forums](https://forums.ootpdevelopments.com/showthread.php?t=220934)
57. [Auto-Calcing the League Totals - OOTP Developments Forums](https://forums.ootpdevelopments.com/showthread.php?p=4253957)
58. [Developing and Maximising Your Youth Intakes in FM26](https://www.footballmanager.com/the-dugout/developing-and-maximising-your-youth-intakes-fm26)
59. [Proving ‘Youth Facilities’ has no effect on newgen PA in Football Manager](https://georgefloydoverdosed.medium.com/proving-youth-facilities-has-no-effect-on-newgen-pa-in-football-manager-819305af2f30)
60. [Newgen mechanics revisited.. Youth Facilities still don't affect PA.](https://fm-arena.com/thread/15782-newgen-mechanics-revisited-youth-facilities-still-don-t-affect-pa/)
61. [FM24 Youth Ratings - Best nations and regions to scout in Football Manager 2024](https://sortitoutsi.net/football-manager-2024-youth-ratings)
62. [Consanguinity, human evolution, and complex diseases](https://www.pnas.org/doi/pdf/10.1073/pnas.0906079106)
