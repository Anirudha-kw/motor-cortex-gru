# Predictions recorded before the data

## 2026-10-20: damping 20, curl +35 N s/m, 6 outward reaches (seed 4), policy s2rR_s1

Written before `mc_d20_session.npz` (null:2, curl:2) was recorded. Source: `expect.cpp` through the real controller
code, with `DAMPING_K=20`, on the nominal simulated arm and on the three plants fitted jointly to the undamped and
damping-12 real reaches (`plant_fit2_top.npy`, top 3).

Signed lateral deviation at peak speed, curl minus null, same reach (mm), outward reaches 1 to 6:

| plant | reach 1 → 6 |
|---|---|
| nominal sim | −1.8 −1.5 −1.3 −2.1 −1.5 −1.9 |
| fit 0 | −2.7 −2.1 −1.7 −1.3 −1.4 −1.5 |
| fit 1 | −2.2 −2.3 −1.7 −2.2 −1.6 −1.6 |
| fit 2 | −2.3 −1.9 −1.6 −1.6 −1.6 −1.7 |

Prediction:
- The curl deflects every outward reach by 1.5–2.7 mm, in the same sign as in simulation.
- The deflection shrinks across the six reaches by at most about half. Without damping (nominal sim) it shrinks
  5×, from −3.5 to −0.7.
- No trips; reach end errors within about 4 mm.

Caveat: these fitted plants under-predicted the real residual vertical swing at damping 12 (3–6 mm simulated against
23 mm real). Real-robot variability in lateral position is about ±1–2 mm, comparable to the effect.

### Outcome (2026-10-20, `out/mc_d20_session.npz`: 2 null + 2 curl episodes, damping 20, real Panda)

Curl minus null at peak speed (mm), outward reaches 1–6 (mean of 2 episodes each):

| | reach 1 | reach 2 | reach 3 | reach 4 | reach 5 | reach 6 |
|---|---|---|---|---|---|---|
| real robot | −2.19 | −1.70 | −1.62 | −2.15 | −1.65 | −0.72 |
| prediction, fit 1 | −2.2 | −2.3 | −1.7 | −2.2 | −1.6 | −1.6 |
| null-to-null spread (noise) | 0.11 | 1.45 | 0.10 | 0.10 | 0.11 | 0.74 |

- **Deflection of 1.5–2.7 mm, same sign as simulation: holds** on reaches 1–5. Reach 6 is −0.7, but reaches 2 and 6
  are the noisy directions (0.7–1.5 mm spread between the two null episodes).
- **Shrinks by at most about half: holds.** Reach 1 / mean of reaches 5–6 = 1.85. On the four clean reaches (1, 3,
  4, 5) the effect is roughly flat at −1.6 to −2.2 mm, i.e. little or no adaptation, as predicted.
- **No trips; end errors within about 4 mm: holds**, with a maximum of 4.7 mm on one null reach.

The simulation with the plant fitted to the earlier real reaches predicts the real reach-by-reach curl effect to
within about 0.5 mm on the clean reaches. The curl effect on the real arm is about 20× the null run-to-run noise on
the clean reaches. With damping 20, the in-context adaptation is not visible on hardware.

## 2026-10-21: balanced session at damping 20 (`--pairs`), policy s2rR_s1. Written before the run.

**Design.**
- N pairs (N ≥ 16) of null and curl episodes, 6 outward reaches each.
- Each pair shares a target order (seed = 100 + k). Null-first and curl-first alternate.

**Statistic.** d_r is curl minus null in signed lateral deviation at peak speed, perpendicular to the movement, at
outward reach r. Average it per reach index over pairs.

**Prediction**, from `balanced.py`: the six plants fitted to the real arm at k = 20 plus the nominal arm, 16 orders
each, through the real controller.

| quantity | predicted |
|---|---|
| d1 | −2.0 to −2.4 mm |
| mean of d5 and d6 | −1.3 to −1.6 mm |
| ratio d1 / mean(d5, d6) | 1.4 to 1.75 |
| decline, d1 − mean(d5, d6) in magnitude | 0.6 to 1.0 mm |
| after-effect | not tested (~0.1 mm predicted, below hardware noise) |

**Hypothesis.** The decline is at least 0.3 mm, with a 95% bootstrap CI over pairs that excludes 0.

**Also expected:** no trips, and end errors within about 5 mm.

### Outcome (2026-10-21, `out/mc_balanced_d20.npz`: 20 pairs, seeds 100–119, damping 20, real Panda)

Scored with `analyze_balanced.py`. Curl minus null at peak speed (mm), mean over 20 pairs [95% bootstrap CI]:

| | reach 1 | reach 2 | reach 3 | reach 4 | reach 5 | reach 6 |
|---|---|---|---|---|---|---|
| real robot | −2.69 [−2.93, −2.47] | −2.01 | −1.26 | −1.97 | −1.56 | −0.86 [−1.28, −0.45] |

| quantity | predicted | real | |
|---|---|---|---|
| d1 | −2.0 to −2.4 | −2.69 | larger than predicted |
| mean of d5 and d6 | −1.3 to −1.6 | −1.21 | slightly smaller than predicted |
| ratio d1 / mean(d5, d6) | 1.4 to 1.75 | 2.23 | larger than predicted |
| decline | 0.6 to 1.0 mm | 1.49 [1.11, 1.81] mm | larger than predicted |

- **Hypothesis (decline ≥ 0.3 mm, CI excludes 0): HOLDS.** All four point predictions missed in the same direction:
  the real arm adapts more than the simulation said, about 1.5–2.5×.
- **No trips: holds** (40/40 episodes status 10). **End errors within about 5 mm: mostly.** Median 2.0 mm, 21 of
  240 reaches over 5 mm, max 11.7 mm. The worst target (s111, reach 1) is bad in both curl (11.7) and null (8.4).
- **The earlier outcome's "adaptation not visible at damping 20" is superseded.** That session used one target order,
  so direction was confounded with reach index (see GATE0.md).

Robustness checks, run after scoring (`balanced_checks.py`; none replaces the primary statistic). Decline [95% CI]:

| check | decline (mm) |
|---|---|
| primary | 1.49 [1.13, 1.81] |
| 5 Hz low-passed hand position (the 12 Hz joint-2 buzz) | 1.38 [1.00, 1.74] |
| direction-adjusted (8 directions) | 1.41 [1.09, 1.72] |
| first / second half of the pairs | 1.30 / 1.67 |
| curl-first / null-first pairs | 1.34 / 1.63 |
| per unit peak speed | 4.34 [3.57, 5.05] mm per m/s |

- Curl and null peak speeds are equal at every reach (difference ≤ 0.006 m/s, CIs include 0), and peak speed does
  not fall over reaches (0.31–0.34 m/s). So the decline is not the arm slowing down in the field.
- What this does not show is *how* the deviation shrinks: a learned feedforward compensation (an internal model)
  or a stiffer response. There were no catch trials, so there is no after-effect test.

## 2026-10-21: catch-trial session at damping 20 (`--pairs 20 --seed 200 --off-after 5`). Written before the run.

**Question.** How does the deviation shrink?
- **Feedforward (internal model):** the GRU pushes against the field before being pushed. This leaves an after-effect
  when the field is unexpectedly off.
- **Not feedforward:** a stiffer arm, or a stronger and faster reaction to the field's push. Either leaves no
  after-effect.

**Evidence so far (simulation, through the real controller; hardware time course).**
- **After-effect:** about 0 in the simulation (+0.01 to +0.10 mm), against a 0.7–1.1 mm decline.
- **Stiffening is small** (`probe_impedance.py`, ±5 N × 50 ms lateral probe, reach 1 vs reach 6). The curl/null
  response ratio falls by −0.00 on the nominal arm and by +0.06, +0.13 and +0.13 on the fitted plants. Only one CI
  excludes 0.
- **Not carry-over from the return reaches.** With the field on only during outward reaches, the decline is
  unchanged: 0.78–1.07 mm against 0.82–1.17 mm.
- **On hardware the shrinkage starts early** (`timecourse.py` on `mc_balanced_d20.npz`). It is significant from
  80 ms after movement onset to 200 ms. Peak speed comes at about 120 ms, and the onset lateral state shows no carry-over.

**Design.** 20 pairs, target orders seeds 200–219 (new orders), null-first and curl-first alternating. In curl
episodes the field is on for outward reaches 1–5 and switched off as reach 6 starts (the experiment script's
`--off-after 5`).

**Prediction** (`catch_predict.py`, the same 20 orders; nominal arm plus fitted plants 0–2):

| quantity | simulation | hardware expectation |
|---|---|---|
| d1 | −2.28 to −2.74 mm | larger (the last session's d1 exceeded the simulation) |
| d5 | −1.43 to −1.64 mm | |
| decline \|d1\| − \|d5\| | 0.67 to 1.10 mm | about 1–1.5 mm (the last session exceeded the simulation 1.5–2.5×) |
| after-effect a = d6, field off | +0.01 to +0.10 mm | about 0, if the hardware mechanism matches the simulation |

**Hypotheses** (`catch_score.py`):
- **H1 (replication):** the decline \|d1\| − \|d5\| is at least 0.3 mm, with a 95% bootstrap CI excluding 0.
- **H2 (mechanism):** φ = a / (\|d1\| − \|d5\|) is the feedforward fraction. A full internal model predicts φ ≈ 1;
  the simulation predicts φ ≈ 0.
  - **Mostly not feedforward** if the 95% CI upper bound of φ is below 0.5.
  - **Substantial feedforward** if its lower bound is above 0.25.
  - **Inconclusive** otherwise.

**Also expected:** no trips. The field switches off between reaches, with the arm near home and nearly still.

### Amendment (2026-10-30, after a reflex in the first attempt, before any outcome was scored)

**What happened.** The first attempt (`out/mc_catch_d20_run1.npz`) stopped in pair 2 (seed 201, curl first). Franka's
Cartesian reflex aborted the motion 0.55 s into the catch reach of the curl episode. The controller's own trips did
not fire. Its episode clock ran on with the arm frozen, so the episode logged as complete. The next episode was
refused ("not at home") and the script stopped. The cause is the known 10–12 Hz joint-2 limit cycle. The external
force the robot estimates swings by ±50 N in x every episode (my reconstruction from `tau_ext_hat_filtered`: per-episode
peaks were 49, 52 and 54 N in this attempt against a median of 47 and a maximum of 55 N over the 40 episodes of
21 October, which had no reflex). The reflex is a tail event at a threshold this rig runs close to, so another one in
40 episodes is likely. Nothing about the controller, policy, damping or thresholds has changed.

**Rule for reflexes** (`catch_score.py` applies it):
- **A reflex voids its pair.** In the logs, a reflexed episode is one in which the arm stands still (handle speed
  under 5 mm/s) for at least 0.2 s while the controller reports running. Normal reaches never stand still for more
  than 0.04 s (all 47 episodes recorded at damping 20).
- **Resume, don't restart.** After a reflex, relaunch and continue at the seed after the voided pair, with
  `--pairs` = 20 minus the complete pairs so far. Each segment goes to its own file. Replacement seeds run past 219.
- **Scoring uses the first 20 complete, non-voided pairs** in seed order, pooled across the segment files.
- **Stop after 3 reflexes.** The session is then not feasible at these thresholds, and the next step is a decision,
  not another segment.
- **Reported alongside the result:** the voided pairs and the reach at which each reflex hit. A reflex that
  preferentially hits catch reaches could bias the after-effect, because it would remove the largest catch movements.

**For this attempt:** pair seed 200 is complete and counts. Pair seed 201 is voided. The resume is `--seed 202
--pairs 19` (seeds 202–220). Seed 202 starts null-first, as it would have in the original order. The replacement
seed 220 is also null-first, so the order split is 11 null-first and 9 curl-first instead of 10 and 10.

### Amendment 2 (2026-10-30, after the second reflex, before any outcome was scored)

**Second reflex.** Segment 2 (`out/mc_catch_d20_run2.npz`, `--seed 202 --pairs 19`) completed seeds 202–208. A Cartesian
reflex then hit in seed 209 (curl first), 0.23 s after reach index 10 began. That is the start of the catch reach, with
the hand still at home, during the same 10 Hz joint-2 limit cycle. Complete pairs: 200, 202–208 (8). Voided: 201,
209. Both reflexes came at reach index 10 of a curl episode.

**The arm wobbles more today than on 21 October.** Measured on the full 1 kHz log of each complete episode, the 7–15 Hz
share of joint-2 velocity is 0.71 (null) and 0.67 (curl) today, against 0.51 and 0.53 on 21 October. It is high from
the first episode of the session. The warm-up episode, about four minutes before the session, read 0.52. The direction mix of the reaches is
the same (diagonals 50% on both days), so the target orders don't explain it. The gravity model and the resting
external torque at home match 21 October to within 0.15 N·m, so the tool and load settings haven't changed. The
cause is unknown. Joint friction or temperature are the likely candidates. **Caveat for H1:** the replication runs on
an arm whose limit cycle is stronger than in the session it replicates.

**Next segment.** `--seed 210 --pairs 12` (seeds 210–221). Seed 210 is null-first, as in the original order. One of two
operator choices is made before it starts and recorded here:
- **(a) Thresholds unchanged.** The 3-reflex stop applies: a third reflex ends the session with the pairs completed by
  then.
- **(b) Cartesian force reflex thresholds raised from 40 to 60 N** (x, y, z; moments stay at 50 N·m), as the runbook's
  setup section allows for reflexes on normal reaches. This changes nothing below the threshold, so the controller and
  the measurement are unchanged. It does not loosen the joint-torque reflexes or the controller's own trips. The
  3-reflex stop still applies.

Choice made: _(to be filled in before the segment starts)_
