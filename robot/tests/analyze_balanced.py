"""Score a balanced session (motor_cortex_experiment.py --pairs) against PREDICTIONS.md (2026-10-21).

d_r = curl - null signed lateral deviation at peak speed, perpendicular to the movement, outward reach r, per pair
(the pair's two episodes share the target order).  Reports the mean d_r over pairs with bootstrap CIs, the decline
d1 - mean(d5, d6) (in magnitude) with its 95% CI, and the ratio.

    python3 analyze_balanced.py ~/mc_balanced_d20.npz
"""
import re
import sys

import numpy as np


def lateral_at_peak(tel):
    run = tel[tel[:, 1] == 2]
    home = run[run[:, 3] == 1][-1, 6:9]                       # the first return target = home
    out, ends = [], []
    for r in range(0, int(run[:, 3].max()) + 1, 2):
        rr = run[run[:, 3] == r]
        tgt = rr[-1, 6:9]
        d = (tgt - home)[:2]
        if len(rr) < 10 or np.linalg.norm(d) < 1e-3:          # the rows after the last reach, not a reach
            continue
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        sp = np.linalg.norm(rr[:, 12:14], axis=1)
        i = int(sp.argmax())
        out.append(1e3 * (rr[i, 9:11] - home[:2]) @ n)
        ends.append(1e3 * np.linalg.norm(rr[-1, 9:12] - tgt))
    return np.array(out), np.array(ends)


def main():
    z = np.load(sys.argv[1])
    eps = {}
    for k in z.files:
        m = re.match(r"episode_\d+_(null|curl)_s(\d+)$", k)
        if m:
            eps.setdefault(int(m.group(2)), {})[m.group(1)] = z[k]
    pairs = sorted(s for s, e in eps.items() if "null" in e and "curl" in e)
    D, ends = [], []
    for s in pairs:
        c, ec = lateral_at_peak(eps[s]["curl"])
        n, en = lateral_at_peak(eps[s]["null"])
        D.append(c - n)
        ends += list(ec) + list(en)
    D = np.array(D)                                           # (pairs, 6)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(D), (10000, len(D)))
    boot = D[idx].mean(1)                                     # (10000, 6)
    mean = D.mean(0)
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    decline = np.abs(boot[:, 0]) - np.abs(boot[:, 4:6].mean(1))
    dec = abs(mean[0]) - abs(mean[4:6].mean())
    ratio = mean[0] / mean[4:6].mean()
    print(f"{len(pairs)} complete pairs; end error max {max(ends):.1f} mm")
    print("curl - null per outward reach (mm), mean [95% CI]:")
    for r in range(D.shape[1]):
        print(f"   reach {r + 1}: {mean[r]:+.2f} [{lo[r]:+.2f}, {hi[r]:+.2f}]")
    print(f"decline |d1| - |mean(d5, d6)| = {dec:.2f} mm [{np.percentile(decline, 2.5):.2f}, "
          f"{np.percentile(decline, 97.5):.2f}];  ratio d1 / mean(d5, d6) = {ratio:.2f}")
    ok = dec >= 0.3 and np.percentile(decline, 2.5) > 0
    print("prediction (PREDICTIONS.md 2026-10-21): d1 -2.0..-2.4, d5/6 -1.3..-1.6, ratio 1.4-1.75, decline 0.6-1.0;"
          f" hypothesis (decline >= 0.3 mm, CI excludes 0): {'HOLDS' if ok else 'FAILS'}")


if __name__ == "__main__":
    main()
