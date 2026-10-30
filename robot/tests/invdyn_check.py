"""Inverse dynamics on the real motion (1 kHz log): the torque the rigid-body model needs to produce the measured
q, dq, ddq, against the torque the sensors measured (tau_J - gravity) and the torque commanded (tau_J_d).  No
integration, so no drift.  Velocities and accelerations are smoothed with the same zero-phase filter on all signals.

    python3 invdyn_check.py ~/mc_reach_logged.npz
"""
import sys

import mujoco
import numpy as np

from replay_measured import DQ, G, Q, TD, TJ, model, uniform


def smooth(x, hz=40.0):
    """Zero-phase 2nd-order Butterworth-like smoothing (forward-backward exponential twice) at about `hz`."""
    a = 1 - np.exp(-2 * np.pi * hz * 0.001)
    y = x.copy()
    for _ in range(2):
        for i in range(1, len(y)):
            y[i] = y[i - 1] + a * (y[i] - y[i - 1])
        for i in range(len(y) - 2, -1, -1):
            y[i] = y[i + 1] + a * (y[i] - y[i + 1])
    return y


def main():
    z = np.load(sys.argv[1])
    key = [k for k in z.files if k.endswith("_fast") and k.startswith("episode_")][0]
    f = z[key]
    t, f = uniform(f[f[:, 1] == 2])
    q, dq = f[:, Q], smooth(f[:, DQ])
    ddq = np.gradient(dq, 0.001, axis=0)
    meas = smooth(f[:, TJ] - f[:, G])
    cmd = smooth(f[:, TD])
    meas -= meas[:20].mean(0)          # sensor and gravity-model offsets: the arm starts at rest
    cmd -= cmd[:20].mean(0)
    print(f"{key}: {len(t)} ms running")
    print("rms (N m) per joint of   inverse dynamics - measured   |   inverse dynamics - commanded   (cmd rms for scale)")
    for arm, tool in ((0.0, 0.63), (0.0, 0.0), (0.05, 0.63), (0.1, 0.63), (0.3, 0.63)):
        m = model(arm, tool)
        d = mujoco.MjData(m)
        tau = np.zeros_like(q)
        for k in range(len(t)):
            d.qpos[:7], d.qvel[:7], d.qacc[:7] = q[k], dq[k], ddq[k]
            mujoco.mj_inverse(m, d)
            tau[k] = d.qfrc_inverse[:7]
            d.qvel[:7], d.qacc[:7] = 0.0, 0.0
            mujoco.mj_inverse(m, d)
            tau[k] -= d.qfrc_inverse[:7]          # remove gravity: compare with gravity-free torques
        tau -= tau[:20].mean(0)
        em = np.sqrt(((tau - meas) ** 2).mean(0))
        ec = np.sqrt(((tau - cmd) ** 2).mean(0))
        print(f"  armature {arm:<4} tool {tool:<4}  {np.round(em, 2)}  |  {np.round(ec, 2)}")
        if arm == 0.0 and tool == 0.63:
            best = tau
    print(f"  cmd rms                       {np.round(np.sqrt((cmd ** 2).mean(0)), 2)}")
    print("\nj2 and j4 every 20 ms, armature 0, tool 0.63: inverse dynamics / measured / commanded (N m)")
    for k in range(0, len(t), 20):
        print(f"  {1e3 * t[k]:4.0f}  j2 {best[k, 1]:+6.1f} {meas[k, 1]:+6.1f} {cmd[k, 1]:+6.1f}   "
              f"j4 {best[k, 3]:+6.1f} {meas[k, 3]:+6.1f} {cmd[k, 3]:+6.1f}   j6 {best[k, 5]:+5.1f} {meas[k, 5]:+5.1f} {cmd[k, 5]:+5.1f}")


if __name__ == "__main__":
    main()
