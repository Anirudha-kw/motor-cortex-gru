"""Random search for the plant (plant_probe.run) whose closed loop with the policy reproduces the real episode, scored
on what the policy saw each tick: handle height and the velocities of joints 2, 4 and 6.

    python3 plant_fit.py ../../sim/out/mc_reach_logged.npz [n]
"""
import sys

import numpy as np

from plant_probe import run

J = [1, 3, 5]


def score(seen, real, n):
    ez = 1e3 * (seen[:n, 2] - real[:n, 2]) / 5                         # handle height [mm]
    edq = (seen[:n, 10 + np.array(J)] - real[:n, 10 + np.array(J)]) / 0.2   # joint velocity [rad/s]
    return np.sqrt((ez ** 2).mean()) / 10 + np.sqrt((edq ** 2).mean()) / 0.3, np.sqrt((ez ** 2).mean())


def main():
    z = np.load(sys.argv[1])
    p = z["episode_001_null_policy"]
    real = np.c_[p[:, 8:11], p[:, 11:18], p[:, 18:25]]                  # handle*5, q - q_home, dq*0.2 (10 ms old)
    n = len(real)
    rng = np.random.default_rng(0)
    res = []
    for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 600):
        gain = rng.uniform(0.5, 1.05, 7)
        lag = np.exp(rng.uniform(np.log(0.002), np.log(0.05), 7))
        fric = rng.uniform(0, 2.0, 7) * np.array([1, 1, 1, 1, .3, .5, .3])
        arm = float(np.exp(rng.uniform(np.log(0.02), np.log(0.4))))
        pl = dict(armature=arm, gain=gain, lag=lag, friction=fric)
        _, trip, dqmax, seen = run(pl, n_ticks=n)
        sc, rz = score(seen, real, n)
        res.append((sc, rz, trip, pl))
    res.sort(key=lambda r: r[0])
    zr = 1e3 * real[:, 2] / 5
    print("real height every 30 ms:", " ".join(f"{zr[i]:+4.0f}" for i in range(2, n, 3)))
    for sc, rz, trip, pl in res[:8]:
        _, _, _, seen = run(pl, n_ticks=n)
        print(f"score {sc:.2f} (z rms {rz:4.1f} mm) trip {trip}  armature {pl['armature']:.3f}")
        print(f"   gain {np.round(pl['gain'], 2)}  lag ms {np.round(1e3 * pl['lag'], 0)}  fric {np.round(pl['friction'], 1)}")
        print("   sim height:", " ".join(f"{1e3 * seen[i, 2] / 5:+4.0f}" for i in range(2, n, 3)))
    np.save("plant_fit_top.npy", np.array([dict(score=r[0], **r[3]) for r in res[:20]], dtype=object))


if __name__ == "__main__":
    main()
