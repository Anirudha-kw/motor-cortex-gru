# Gate 0: is there headroom for a learned impedance schedule?

Written 2026-10-20, before running `gate0.py`.

## Question

The deployed policy (`s2rR_s1`) needs joint damping to be stable on the real arm. That damping suppresses its
in-context curl adaptation. Does some fixed damping give both stability and adaptation? If so, a learned
impedance schedule has nothing to add, and the idea dies.

## Setup

- **Controller:** the real controller source (`expect.cpp`), at 1 kHz, with first-session caps of 20/5 N m.
- **Policy:** fixed weights, `s2rR_s1`. Nothing is trained.
- **Plants:** the six fitted jointly to the undamped and damping-12 real reaches (`plant_fit2_top.npy`, top 6).
  Each has per-joint torque gain, lag and friction, an armature, and a 0.63 kg tool.
- **Episodes,** each with the same 6 outward reaches (seed 4):
  - A: null field;
  - B: curl +35 N s/m;
  - C: curl, switched off after the 3rd outward reach.
- **Damping:**
  - fixed k ∈ {0, 4, 8, 12, 16, 20, 25, 30} N m s/rad (joints 1–4; ×0.2 on 5–6, ×0.1 on 7);
  - three hand-set schedules:
    - `decay`: k = 20 on the first out-and-back, 12 on the second, 6 after;
    - `within`: k = 4 from 0.1 to 0.4 s of each reach (the movement), 20 otherwise;
    - `within_rev`: the reverse, as a control.

## Metrics, per plant

- **Stable:** all three episodes end with status 10 (no trip), and the handle stays within 10 mm of the home
  height throughout.
- **Adaptation ratio A** = |d1| / mean(|d5|, |d6|). Here d_r is the signed lateral deviation at peak speed, curl
  minus null, on outward reach r of episode B. Reference: the nominal simulated arm without damping gives A ≈ 4.
- **After-effect:** (C − null) on reach 4, the first reach with the field off. It counts if its sign is opposite
  to d1.
- **Effort:** the mean |commanded torque| on joints 1–4.

## Decision

A setting **passes** if, on all six plants, it is stable and A ≥ 2.

- **Headroom for a learned schedule exists** if no fixed k passes.
- Fixed k = 12 counts as unstable whatever the simulation says. On the real arm it swung 23 mm against the 10 mm
  limit, while these plants under-predicted that swing at 3–6 mm. So the simulation's stability verdict is
  optimistic.
- If no fixed k passes, but a schedule does, that is direct evidence that time-varying impedance solves the
  trade-off with this policy. Proceed with learned impedance.
- If some fixed k passes, stop: deploy that k and drop the idea.
- If nothing passes, fixed or scheduled, scheduling impedance alone does not resolve the trade-off for this
  policy. A learned controller would also have to change its feedforward. The idea survives only as a retraining
  question, which is a weaker premise; report it as such.

## Result (2026-10-20, `gate0.py`, `out/gate0.json`)

**Verdict: NEITHER.** No fixed damping passes, and no hand-set schedule passes either. As pre-registered, learned
impedance survives only as a retraining question.

| setting | stable on all 6 fitted plants | min A over the fitted plants | A per fitted plant |
|---|---|---|---|
| 0–8 | no (trips) | – | – |
| 12 | 5/6, and unstable on the real arm | 0.2 | 2.1 1.7 2.2 1.2 2.4 0.2 |
| 16 | 5/6 | 0.4 | 2.0 1.6 1.5 1.2 1.9 0.4 |
| 20 | yes | 0.8 | 1.9 1.3 1.4 1.1 1.7 0.8 |
| 25 | yes | 1.0 | 1.9 1.8 1.3 1.0 1.6 1.7 |
| 30 | yes | 1.1 | 2.6 1.1 1.2 1.3 1.7 2.9 |
| decay | no | – | – |
| within (low k while moving) | no | – | – |
| within_rev (high k while moving, 4 at rest) | 4/6 | 0.3 | 2.9 1.6 3.3 1.6 1.8 0.3 |

- **Even the nominal simulated arm stops adapting once damped.** A = 4.1 at k = 0, and 1.1–1.9 at every k ≥ 4.
  So the suppression is a property of this policy, not of the real arm's lag. The policy was trained without
  damping in the loop; most likely, damping shrinks the error it adapts on. That is untested.
- **After-effects** have the right sign almost everywhere the plant is stable, but they are tiny (0.05–0.3 mm).
- **Hint, not evidence:** high damping during the movement and low at rest (`within_rev`) gives the largest
  adaptation on two plants (A ≈ 3). It is unstable on two others.

## Next gate (0b)

Re-run this gate on the vY policies (`s2rY_*`), which are trained with the controller's damping randomised in the
loop. If a vY policy at some fixed k passes, retraining alone solves the problem: fixed impedance wins again, and
learned impedance has no headroom. If none passes, the premise for learned impedance holds.

## Gate 0b result (2026-10-20): inconclusive, because the retraining failed

All vY runs stalled once their curriculum widened toward the realistic actuation range. Backpropagating through
the whole 8.4 s episode explodes on plants with long lag and low gain, even with damping of 8 or more:

| run | damping range | stalled at |
|---|---|---|
| s2rY_s1, s2rY_s2 | 0–20 | iteration 300–600 |
| s2rYd4_s1 | 4–20 | at once |
| s2rYd_s1 | 8–20 | iteration ~1500 |
| s2rYd_s2 | 8–20 | iteration ~600 |

The gate was run on the last checkpoints from before the stalls (iteration 1000). They are worse than the
deployed policy on every measure:

| policy | A on the nominal arm, k = 0 | damping at which all 6 fitted plants are stable |
|---|---|---|
| s2rR_s1 (deployed) | 4.1 | 20 and above |
| s2rYd_s1, iteration 1000 | 1.9 | none (5/6 at k = 30) |
| s2rYd_s2, iteration 1000 | 2.0 | none (4/6 at k = 30) |

This is evidence about the training method, not about the premise. Training by backprop through the dynamics
cannot reach the real arm's actuation range. Testing the premise needs a different trainer, for example PPO, which
does not differentiate through the physics, or a much narrower range.

## Correction (2026-10-21): Gate 0's "damping kills adaptation" was a design artefact

Gate 0 used one target order (seed 4), so "reach 1 vs reaches 5–6" also compared different directions.
`balanced.py` reruns it with 16 target orders per cell. For each order, curl and null share the order, so directions
average out. Same controller code, plants and caps (`out/balanced_s2rR_s1.json`).

**Result for the deployed s2rR_s1.** Curl minus null at peak speed falls monotonically across reaches in every stable
cell, by 30–45% over six reaches, at about 10 standard errors.

| plant | damping | reach 1 | reach 6 |
|---|---|---|---|
| nominal | 0 | −2.67 mm | −1.66 mm |
| fit 0 | 20 | −2.31 mm | −1.26 mm |
| fit 4 | 20 | −2.31 mm | −1.43 mm |

- Damping shrinks the curl's effect but keeps the relative adaptation: the reach-1 to reach-5/6 ratio is 1.4–1.75
  at damping 20.
- The after-effect has the right sign but is about 0.1 mm.
- The "A = 4.1 undamped" quoted earlier was inflated by the same confound; the balanced value is 1.56.
- The verdict above ("NEITHER") rested on the confounded metric and is withdrawn.
