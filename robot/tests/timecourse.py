"""Time course of the curl effect within a reach, by reach index (which part of the deviation shrinks?).

Per pair and outward reach: lateral hand position (perpendicular to the movement, from home) in the curl episode
minus the null episode, aligned on movement onset (hand speed first above 0.05 m/s), 100 Hz telemetry.
  same early deviation, smaller late deviation -> faster correction (feedback adaptation)
  smaller from the start                         -> compensation before feedback (feedforward or stiffer arm)

    python3 timecourse.py ~/mc_balanced_d20.npz
"""
import re
import sys

import numpy as np

T = 41                                                        # 0..400 ms after onset


def traces(tel):
    run = tel[tel[:, 1] == 2]
    home = run[run[:, 3] == 1][-1, 6:9]
    out = []
    for r in range(0, int(run[:, 3].max()) + 1, 2):
        rr = run[run[:, 3] == r]
        d = (rr[-1, 6:9] - home)[:2] if len(rr) else np.zeros(2)
        if len(rr) < 10 or np.linalg.norm(d) < 1e-3:
            continue
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        sp = np.linalg.norm(rr[:, 12:14], axis=1)
        i0 = int(np.argmax(sp > 0.05))
        lat = 1e3 * (rr[:, 9:11] - home[:2]) @ n
        seg = lat[i0:i0 + T]
        out.append(np.pad(seg, (0, T - len(seg)), mode="edge"))
    return np.array(out)


def main():
    z = np.load(sys.argv[1])
    eps = {}
    for k in z.files:
        m = re.match(r"episode_\d+_(null|curl)_s(\d+)$", k)
        if m:
            eps.setdefault(int(m.group(2)), {})[m.group(1)] = z[k]
    pairs = sorted(s for s, e in eps.items() if "null" in e and "curl" in e)
    D = np.array([traces(eps[s]["curl"])[:6] - traces(eps[s]["null"])[:6] for s in pairs])   # pairs x 6 x T
    rng = np.random.default_rng(0)
    groups = [("reach 1", [0]), ("reaches 2-3", [1, 2]), ("reach 4", [3]), ("reaches 5-6", [4, 5])]
    ts = [3, 5, 8, 10, 15, 20, 25, 30, 40]
    print(f"{len(pairs)} pairs; curl - null lateral (mm) at ms after movement onset; mean over pairs")
    print("              " + "".join(f"{10 * t:>7d}" for t in ts))
    G = {}
    for name, idx in groups:
        G[name] = D[:, idx, :].mean(1)                        # pairs x T
        print(f"  {name:12s}" + "".join(f"{v:+7.2f}" for v in G[name].mean(0)[ts]))
    diff = np.abs(G["reach 1"]) - np.abs(G["reaches 5-6"])
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean(0) for _ in range(5000)])
    print("  shrinkage    " + "".join(f"{v:+7.2f}" for v in diff.mean(0)[ts]) + "   (|reach 1| - |reaches 5-6|)")
    print("    95% CI lo  " + "".join(f"{v:+7.2f}" for v in np.percentile(b, 2.5, 0)[ts]))
    print("    95% CI hi  " + "".join(f"{v:+7.2f}" for v in np.percentile(b, 97.5, 0)[ts]))
    sp = []
    for s in pairs:
        for c in ("null", "curl"):
            run = eps[s][c][eps[s][c][:, 1] == 2]
            rr = run[run[:, 3] == 0]
            v = np.linalg.norm(rr[:, 12:14], axis=1)
            sp.append(np.argmax(v) - np.argmax(v > 0.05))
    print(f"  (peak speed is reached ~{10 * np.median(sp):.0f} ms after onset on reach 1)")


if __name__ == "__main__":
    main()
