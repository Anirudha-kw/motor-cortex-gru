"""Robustness checks on a balanced session, after the pre-registered score (analyze_balanced.py).  None of these
replaces the primary statistic; each asks whether something other than within-episode adaptation could make
|d1| > |d5, d6|.

  speed      peak hand speed per reach index, curl and null (the curl force scales with speed)
  per-speed  d_r divided by the pair's mean peak speed at that reach (mm per m/s)
  lowpass    hand position low-passed at 5 Hz before sampling at peak speed (the 12 Hz buzz)
  direction  d_r minus the mean d of its movement direction over all reach indices (direction balance)
  halves     first vs second half of the pairs (drift over the session); curl-first vs null-first pairs (order)
  end error  per condition

    python3 balanced_checks.py ~/mc_balanced_d20.npz
"""
import re
import sys

import numpy as np


def lowpass(x, fc, dt):
    X = np.fft.rfft(x - x[0], axis=0)
    f = np.fft.rfftfreq(len(x), dt)
    X *= 1.0 / (1.0 + (f / fc) ** 8)[:, None]                 # zero-phase, 4th-order-like roll-off
    return np.fft.irfft(X, n=len(x), axis=0) + x[0]


def reaches(tel, smooth=False):
    """Per outward reach: lateral at peak speed (mm), peak speed (m/s), direction angle (deg), end error (mm)."""
    run = tel[tel[:, 1] == 2]
    pos = lowpass(run[:, 9:12], 5.0, 0.01) if smooth else run[:, 9:12]
    home = run[run[:, 3] == 1][-1, 6:9]
    out = []
    for r in range(0, int(run[:, 3].max()) + 1, 2):
        m = run[:, 3] == r
        rr, pp = run[m], pos[m]
        tgt = rr[-1, 6:9]
        d = (tgt - home)[:2]
        if len(rr) < 10 or np.linalg.norm(d) < 1e-3:
            continue
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        sp = np.linalg.norm(rr[:, 12:14], axis=1)
        i = int(sp.argmax())
        out.append((1e3 * (pp[i, :2] - home[:2]) @ n, sp[i], np.degrees(np.arctan2(d[1], d[0])),
                    1e3 * np.linalg.norm(rr[-1, 9:12] - tgt)))
    return np.array(out)


def ci(x, rng):
    b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def decline(D, rng):
    b = D[rng.integers(0, len(D), (10000, len(D)))].mean(1)
    stat = np.abs(b[:, 0]) - np.abs(b[:, 4:6].mean(1))
    m = D.mean(0)
    return abs(m[0]) - abs(m[4:6].mean()), np.percentile(stat, 2.5), np.percentile(stat, 97.5)


def main():
    z = np.load(sys.argv[1])
    eps, first = {}, {}
    for k in z.files:
        m = re.match(r"episode_(\d+)_(null|curl)_s(\d+)$", k)
        if m:
            s = int(m.group(3))
            eps.setdefault(s, {})[m.group(2)] = z[k]
            if s not in first or int(m.group(1)) < first[s][0]:
                first[s] = (int(m.group(1)), m.group(2))
    pairs = sorted(s for s, e in eps.items() if "null" in e and "curl" in e)
    rng = np.random.default_rng(0)
    R = {(s, c, sm): reaches(eps[s][c], sm) for s in pairs for c in ("null", "curl") for sm in (False, True)}

    def D(sm=False, sel=pairs):
        return np.array([R[(s, "curl", sm)][:6, 0] - R[(s, "null", sm)][:6, 0] for s in sel])

    print(f"{len(pairs)} pairs\n")
    print("peak hand speed (m/s) per outward reach, mean over pairs")
    for c in ("null", "curl"):
        v = np.array([R[(s, c, False)][:6, 1] for s in pairs])
        print(f"  {c:5s} " + " ".join(f"{x:.3f}" for x in v.mean(0)))
    v = np.array([R[(s, "curl", False)][:6, 1] - R[(s, "null", False)][:6, 1] for s in pairs])
    lo, hi = ci(v, rng)
    print("  curl-null " + " ".join(f"{a:+.3f}[{b:+.3f},{c:+.3f}]" for a, b, c in zip(v.mean(0), lo, hi)))

    rows = [("primary (as registered)", D())]
    sp = np.array([(R[(s, "curl", False)][:6, 1] + R[(s, "null", False)][:6, 1]) / 2 for s in pairs])
    rows.append(("per unit speed (mm per m/s)", D() / sp))
    rows.append(("5 Hz low-passed position", D(True)))
    # direction adjustment: mean d per direction bin over every reach index, subtracted
    ang = np.array([np.round(R[(s, "curl", False)][:6, 2] / 15) * 15 for s in pairs])
    Dp = D()
    adj = Dp.copy()
    for a in np.unique(ang):
        adj[ang == a] -= Dp[ang == a].mean()
    adj += Dp.mean()                                          # keep the overall level, remove direction structure
    rows.append((f"direction-adjusted ({len(np.unique(ang))} directions)", adj))
    half = len(pairs) // 2
    rows.append(("first half of pairs", D(sel=pairs[:half])))
    rows.append(("second half of pairs", D(sel=pairs[half:])))
    rows.append(("curl-first pairs", D(sel=[s for s in pairs if first[s][1] == "curl"])))
    rows.append(("null-first pairs", D(sel=[s for s in pairs if first[s][1] == "null"])))
    print("\ncheck                                d1      d2      d3      d4      d5      d6     decline [95% CI]   ratio")
    for name, X in rows:
        m = X.mean(0)
        dec, lo, hi = decline(X, rng)
        print(f"  {name:34s}" + "".join(f"{v:+7.2f} " for v in m)
              + f"  {dec:+.2f} [{lo:+.2f}, {hi:+.2f}]   {m[0] / m[4:6].mean():.2f}   (n={len(X)})")

    print("\nend error (mm): median / 95th pct / max, and reaches over 5 mm")
    for c in ("null", "curl"):
        e = np.concatenate([R[(s, c, False)][:, 3] for s in pairs])
        print(f"  {c:5s} {np.median(e):5.1f} / {np.percentile(e, 95):5.1f} / {e.max():5.1f}   {np.sum(e > 5)}/{len(e)}")
    worst = sorted(((R[(s, c, False)][i, 3], s, c, i + 1) for s in pairs for c in ("null", "curl")
                    for i in range(len(R[(s, c, False)]))), reverse=True)[:5]
    print("  worst: " + ", ".join(f"{e:.1f} mm (s{s} {c} reach {i})" for e, s, c, i in worst))


if __name__ == "__main__":
    main()
