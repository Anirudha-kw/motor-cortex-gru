"""plant_fit.py over several logged real episodes at once, each run with the controller damping it had:

    python3 plant_fit2.py n seed out.npy ../../sim/out/mc_reach_logged.npz:0 ../../sim/out/mc_damped_1.npz:12
"""
import sys

import numpy as np

from plant_fit import score
from plant_probe import run

SHAPE = np.array([1, 1, 1, 1, 0.2, 0.2, 0.1])


def load(spec):
    path, k = spec.rsplit(":", 1)
    p = np.load(path)["episode_001_null_policy"]
    return np.c_[p[:, 8:11], p[:, 11:18], p[:, 18:25]], float(k)


def evaluate(pl, episodes):
    total, zs, trips = 0.0, [], []
    for real, k in episodes:
        n = len(real)
        _, trip, _, seen = run({**pl, "ctrl_damping": k * SHAPE}, n_ticks=n)
        sc, rz = score(seen, real, n)
        total += sc
        zs.append(rz)
        trips.append(trip)
    return total, zs, trips


def main():
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    episodes = [load(s) for s in sys.argv[4:]]
    rng = np.random.default_rng(seed)
    res = []
    for i in range(n):
        pl = dict(gain=rng.uniform(0.4, 1.05, 7), lag=np.exp(rng.uniform(np.log(0.002), np.log(0.06), 7)),
                  friction=rng.uniform(0, 2.0, 7) * np.array([1, 1, 1, 1, .3, .5, .3]),
                  armature=float(np.exp(rng.uniform(np.log(0.02), np.log(0.4)))))
        res.append((*evaluate(pl, episodes), pl))
    res.sort(key=lambda r: r[0])
    np.save(out, np.array([dict(score=r[0], z_rms=r[1], trips=r[2], **r[3]) for r in res[:30]], dtype=object))


if __name__ == "__main__":
    main()
