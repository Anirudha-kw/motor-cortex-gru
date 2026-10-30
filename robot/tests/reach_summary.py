"""One-line-per-file summary of logged real reaches (motor_cortex_experiment.py --record, controller v2+).

    python3 reach_summary.py a.npz b.npz ...
"""
import sys

import numpy as np


def summary(path):
    z = np.load(path)
    for key in [k for k in z.files if k.startswith("episode_") and not k.endswith(("_policy", "_fast"))]:
        tel, p, f = z[key], z[key + "_policy"], z[key + "_fast"]
        run = tel[tel[:, 1] == 2]
        home = run[0, 9:12]
        fr = f[f[:, 1] == 2]
        d = run[:, 9:12] - home
        n_reach = int(run[:, 3].max()) + 1
        ends = []
        for r in range(n_reach):
            rr = run[run[:, 3] == r]
            ends.append(1e3 * np.linalg.norm(rr[-1, 9:12] - rr[-1, 6:9]))
        zr = 1e3 * np.abs(d[30:, 2]).max() if len(d) > 30 else np.nan
        capped = (np.abs(p[:, 39:43]) >= 19.99).any(1).mean()
        tripped = (tel[:, 1] == 3).any()
        print(f"{path.split('/')[-1]:24s} {key:22s} {len(run) / 100:4.2f} s {'TRIP' if tripped else 'done'} | "
              f"end errors (mm) {np.round(ends, 1)} | max |z| after 0.3 s {zr:5.1f} mm | "
              f"peak |dq| {np.abs(fr[:, 9:16]).max():.2f} | policy at cap {capped:.0%} | "
              f"peak speed {np.linalg.norm(run[:, 12:14], axis=1).max():.2f} m/s")


for path in sys.argv[1:]:
    summary(path)
