"""Damping levels on the plants fitted to BOTH real runs (plant_fit2_top.npy): residual vertical swing after 0.3 s,
share of policy ticks at the cap, the outward reach's end x, peak joint speed; 2 out-and-back reaches (2.8 s)."""
import numpy as np

from plant_probe import run

SHAPE = np.array([1, 1, 1, 1, 0.2, 0.2, 0.1])
top = [{k: v for k, v in r.items() if k not in ("score", "z_rms", "trips")}
       for r in np.load("plant_fit2_top.npy", allow_pickle=True)[:8]]
print("damping | per plant: TRIP t, or [max |z| after 0.3 s (mm), end x of reach 1 (mm), peak |dq|]   | worst |z|")
for k in (0, 12, 16, 20, 25, 30):
    cells, worst = [], 0.0
    for pl in top:
        z, trip, dqmax, seen = run({**pl, "ctrl_damping": k * SHAPE}, n_ticks=280)
        if trip:
            cells.append(f"TRIP {trip:.2f}     ")
            worst = np.inf
        else:
            zz = np.abs(z[30:]).max()
            worst = max(worst, zz)
            cells.append(f"{zz:4.1f} {1e3 * run.xs[69, 0]:3.0f} {dqmax.max():.2f}")
    print(f"  {k:3d}   | " + " | ".join(cells) + f" | {worst:.1f}")
