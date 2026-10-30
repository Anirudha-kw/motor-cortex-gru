"""Compare a real-robot telemetry recording (motor_cortex_experiment.py --record) with the simulated run of the same
controller (record_expect's log): joint speeds, hand speed, torques, and where the real run tripped.

    python3 compare_real.py ~/mc_first_reach.npz expect_log.txt [out.png]
Telemetry columns: 0 t, 1 mode, 2 event, 3 reach, 4 step, 5 go, 6-8 target, 9-11 hand, 12-14 hand velocity,
15-21 q, 22-28 dq, 29-35 commanded torque, 36-42 policy output u, 43-45 field force.
"""
import sys

import numpy as np

DQ_TRIP = np.array([1.85, 1.85, 1.85, 1.85, 2.2, 2.2, 2.2])


def first_reach(rows):
    run = rows[rows[:, 1] == 2]
    return run


def main():
    z = np.load(sys.argv[1])
    real = z[[k for k in z.files if k.startswith("episode_") and not k.endswith(("_policy", "_fast"))][0]
          if any(k.startswith("episode_") for k in z.files) else z.files[0]]
    sim_log = np.loadtxt(sys.argv[2])
    sim = sim_log[(sim_log[:, 1] == 2)][:, 3:]          # episode 1 (no field) of the simulated run, telemetry cols
    sim = sim[sim[:, 1] == 2]
    real_run = real[real[:, 1] == 2]
    after = real[np.searchsorted(real[:, 0], real_run[-1, 0]) + 1:] if len(real_run) else real[:0]
    print(f"real: {len(real_run)} running rows (100 Hz) = {len(real_run) / 100:.2f} s before the trip; "
          f"event codes seen {sorted(set(real[:, 2].astype(int)))}")
    n = min(len(real_run), len(sim))
    t = np.arange(n) / 100
    print("\n  t [s]   go   |dq| real (per joint, rad/s)                        |dq| sim                                      "
          "hand speed real / sim")
    for k in range(n):
        print(f"  {t[k]:5.2f}  {int(real_run[k, 5])}   {np.round(np.abs(real_run[k, 22:29]), 2)}   "
              f"{np.round(np.abs(sim[k, 22:29]), 2)}   {np.linalg.norm(real_run[k, 12:14]):.2f} / {np.linalg.norm(sim[k, 12:14]):.2f}")
    worst = np.abs(real_run[:, 22:29]).max(0)
    print("\npeak |dq| real:", np.round(worst, 2), " trip:", DQ_TRIP, " joints over:", list(np.where(worst > DQ_TRIP)[0] + 1))
    print("peak |dq| sim (same reach):", np.round(np.abs(sim[:n + 30, 22:29]).max(0), 2))
    print("peak |torque| real:", np.round(np.abs(real_run[:, 29:36]).max(0), 1))
    print("peak |torque| sim: ", np.round(np.abs(sim[:n + 30, 29:36]).max(0), 1))
    if len(after):
        print(f"after the trip: {len(after)} rows; |dq| max at the end {np.round(np.abs(after[-1, 22:29]), 3)}")
    if len(sys.argv) > 3:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        m = min(len(sim), n + 30)
        for j in range(7):
            axes[0].plot(np.arange(len(real_run)) / 100, real_run[:, 22 + j], label=f"j{j + 1} real")
            axes[1].plot(np.arange(m) / 100, sim[:m, 22 + j], label=f"j{j + 1} sim")
        for ax, title in zip(axes, ("real robot", "simulation, same reach")):
            ax.axhline(1.85, color="k", lw=0.5, ls=":"); ax.axhline(-1.85, color="k", lw=0.5, ls=":")
            ax.set_ylabel("dq [rad/s]"); ax.set_title(title, fontsize=9, loc="left")
        axes[0].legend(ncol=4, fontsize=7)
        axes[1].set_xlabel("time from episode start [s]")
        fig.tight_layout()
        fig.savefig(sys.argv[3], dpi=150)
        print("wrote", sys.argv[3])


if __name__ == "__main__":
    main()
