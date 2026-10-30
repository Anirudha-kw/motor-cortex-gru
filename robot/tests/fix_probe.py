"""Controller-side fixes on the plants that reproduce the real failure (plant_fit_top.npy): does the growing vertical
oscillation stop, and does the reach still land?  Two out-and-back reaches (2.8 s) per plant."""
import numpy as np

from plant_probe import run

top = list(np.load("plant_fit_top.npy", allow_pickle=True))[:6]
D = np.array([1, 1, 1, 1, 0.2, 0.2, 0.1])
fixes = {"none": {}}
for k in (2, 4, 8, 15):
    fixes[f"damping {k}"] = dict(ctrl_damping=k * D)
for hz in (15, 8, 4):
    fixes[f"lowpass {hz} Hz"] = dict(ctrl_lowpass=1 - np.exp(-2 * np.pi * hz * 0.001))
print("fix            | per plant: trip time or peak |z| after 0.7 s (mm) / x at end of outward reach (mm, target 60)")
for name, fx in fixes.items():
    cells = []
    for pl in top:
        pl = {k: v for k, v in pl.items() if k != "score"}
        z, trip, dqmax, _ = run({**pl, **fx}, n_ticks=280)
        xs = run.xs
        cells.append(f"{'TRIP %.2f' % trip if trip else '%5.1f' % np.abs(z[70:]).max():>9s}/{1e3 * xs[69, 0]:4.0f}")
    print(f"{name:14s} | " + "  ".join(cells))
