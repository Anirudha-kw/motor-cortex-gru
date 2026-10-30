"""Score a catch-trial session (motor_cortex_experiment.py --pairs N --off-after 5) against PREDICTIONS.md.

Curl episodes have the field on for outward reaches 1-5 and off on reach 6.  d_r = curl - null signed lateral
deviation at peak speed (as analyze_balanced.py), per pair.
  H1 (replication)  decline |d1| - |d5| >= 0.3 mm, 95% bootstrap CI excluding 0
  H2 (mechanism)    after-effect a = d6 (field off); feedforward fraction phi = a / (|d1| - |d5|)

A Franka reflex voids its pair (PREDICTIONS.md, amendment of 2026-10-30): the session resumes in a new file, and
the first 20 complete, non-voided pairs in seed order across all the files are scored.

    python3 catch_score.py ~/mc_catch_d20_run1.npz ~/mc_catch_d20_run2.npz ...
"""
import re
import sys

import numpy as np

from analyze_balanced import lateral_at_peak


N_PAIRS = 20


def reflexed(tel, still=0.005, dur=0.2):
    """A reflex freezes the arm while the controller's episode clock runs on: at least `dur` s of status 2 (running)
    with the handle speed under `still` m/s.  Normal reaches never stand still for more than 0.04 s.  Returns the
    reach index at which the stall started, or None."""
    slow = (tel[:, 1] == 2) & (np.linalg.norm(tel[:, 12:15], axis=1) < still)
    n = 0
    for i, s in enumerate(slow):
        n = n + 1 if s else 0
        if n * 0.01 >= dur:
            return int(tel[i - n + 1, 3])
    return None


def main():
    eps, voided = {}, {}
    for path in sys.argv[1:]:
        z = np.load(path)
        for k in z.files:
            m = re.match(r"episode_\d+_(null|curl)_s(\d+)$", k)
            if not m:
                continue
            seed = int(m.group(2))
            r = reflexed(z[k])
            if r is not None:
                voided[seed] = f"{m.group(1)} episode, reach index {r} ({path})"
            elif seed not in voided:
                eps.setdefault(seed, {}).setdefault(m.group(1), z[k])
    for seed, where in sorted(voided.items()):
        print(f"seed {seed} voided: reflex in the {where}")
    pairs = sorted(s for s, e in eps.items() if "null" in e and "curl" in e and s not in voided)[:N_PAIRS]
    if len(pairs) < N_PAIRS:
        print(f"only {len(pairs)} complete pairs of the {N_PAIRS} registered: provisional")
    D = np.array([lateral_at_peak(eps[s]["curl"])[0][:6] - lateral_at_peak(eps[s]["null"])[0][:6] for s in pairs])
    rng = np.random.default_rng(0)
    B = D[rng.integers(0, len(D), (10000, len(D)))].mean(1)
    lo, hi = np.percentile(B, [2.5, 97.5], 0)
    m = D.mean(0)
    print(f"{len(pairs)} pairs; curl - null at peak speed (mm), mean [95% CI]; reach 6 = field off in the curl episode")
    for r in range(6):
        print(f"   reach {r + 1}{' (catch)' if r == 5 else '        '}: {m[r]:+.2f} [{lo[r]:+.2f}, {hi[r]:+.2f}]")
    dec_b = np.abs(B[:, 0]) - np.abs(B[:, 4])
    dec = abs(m[0]) - abs(m[4])
    ok1 = dec >= 0.3 and np.percentile(dec_b, 2.5) > 0
    print(f"H1 decline |d1| - |d5| = {dec:+.2f} [{np.percentile(dec_b, 2.5):+.2f}, {np.percentile(dec_b, 97.5):+.2f}]: "
          f"{'HOLDS' if ok1 else 'FAILS'}")
    phi_b = B[:, 5] / np.where(np.abs(dec_b) > 1e-6, dec_b, np.nan)
    plo, phi_hi = np.nanpercentile(phi_b, [2.5, 97.5])
    phi = m[5] / dec if abs(dec) > 1e-6 else float("nan")
    verdict = ("mostly NOT feedforward (phi CI upper < 0.5)" if phi_hi < 0.5 else
               "substantial feedforward (phi CI lower > 0.25)" if plo > 0.25 else "inconclusive")
    print(f"H2 after-effect a = {m[5]:+.2f} mm [{lo[5]:+.2f}, {hi[5]:+.2f}] (sim predicted +0.01..+0.10; a full "
          f"internal model predicts ~ +{dec:.2f});  phi = {phi:+.2f} [{plo:+.2f}, {phi_hi:+.2f}]: {verdict}")


if __name__ == "__main__":
    main()
