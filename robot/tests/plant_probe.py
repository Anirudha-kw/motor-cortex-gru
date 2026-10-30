"""Which simulated plant reproduces the real arm's failure?  The trained policy (numpy, exact) in the deployment loop
(10 ms policy period on the previous period's sensing, caps 20/5 N m, 1 N m/ms rate limit), the same reach as the
logged real episode (6 cm in +x from home, seed 4, one out-and-back), on plants whose torque delivery is per joint:

    delivered = gain * firstorder(cmd, lag) - friction * tanh(dq / 0.05)

on a rigid Menagerie Panda with a given armature and a 0.63 kg tool.  Prints the handle's height every 30 ms next to
the real one, the first joint-speed trip, and the peak joint speeds.

    python3 plant_probe.py ../../sim/out/mc_reach_logged.npz
"""
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
from policy_numpy import GRUPolicy, Q_HOME          # noqa: E402
from s2r_eval import build                          # noqa: E402

CAP = np.array([20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
DQ_TRIP = np.array([1.85, 1.85, 1.85, 1.85, 2.2, 2.2, 2.2])
POLICY = Path(__file__).resolve().parents[2] / "sim/out/meta_gru_s2rR_s1.npz"
REACH, STEPS, GO = 0.06, 70, 10


def run(plant, n_ticks=140):
    true = build(np.ones(7), None, plant.get("damping", 0.0), 0.63, armature=np.full(7, plant["armature"]))
    ctrl = build(tool=0.63, nominal=True)
    d, dc = mujoco.MjData(true), mujoco.MjData(ctrl)
    site = mujoco.mj_name2id(true, mujoco.mjtObj.mjOBJ_SITE, "handle")
    site_c = mujoco.mj_name2id(ctrl, mujoco.mjtObj.mjOBJ_SITE, "handle")
    d.qpos[:7] = Q_HOME
    mujoco.mj_forward(true, d)
    dc.qpos[:7] = Q_HOME
    mujoco.mj_forward(ctrl, dc)
    home = dc.site_xpos[site_c].copy()
    pol = GRUPolicy(str(POLICY), home)
    targets = [home + np.array([REACH, 0, 0]), home.copy(), home + np.array([0, REACH, 0]), home.copy()]
    gain, lag, fric = (np.broadcast_to(np.asarray(plant[k], float), 7) for k in ("gain", "lag", "friction"))
    alpha = 1 - np.exp(-0.001 / np.maximum(lag, 1e-4))
    sent, deliv, filt = np.zeros(7), np.zeros(7), np.zeros(7)
    xs = []
    past = (home.copy(), d.qpos[:7].copy(), np.zeros(7))
    z, trip, dqmax, seen = [], None, np.zeros(7), []
    for k in range(n_ticks):
        r, s = divmod(k, STEPS)
        handle, q, qd = past
        seen.append(np.r_[(handle - home) * 5, q - Q_HOME, qd * 0.2])
        held = np.clip(pol.step(targets[r], float(s >= GO + 1), handle, q, qd), -CAP, CAP)
        dc.qpos[:7] = d.qpos[:7]
        mujoco.mj_kinematics(ctrl, dc)
        past = (dc.site_xpos[site_c].copy(), d.qpos[:7].copy(), d.qvel[:7].copy())
        for _ in range(10):
            ramp = min(1.0, (k * 10 + _ + 1) / 100.0)                             # the controller's 0.1 s ramp-in
            cmd = held - plant.get("ctrl_damping", 0.0) * d.qvel[:7]          # controller-side joint damping
            if plant.get("ctrl_lowpass"):
                filt = filt + plant["ctrl_lowpass"] * (cmd - filt)
                cmd = filt
            if plant.get("coarse"):        # the training simulator: rate limit and lag once per `coarse` ms
                n = int(plant["coarse"])
                if _ % n == 0:
                    sent = sent + np.clip(ramp * np.clip(cmd, -CAP, CAP) - sent, -1.0 * n, 1.0 * n)
                    deliv = deliv + (1 - np.exp(-0.001 * n / np.maximum(lag, 1e-4))) * (sent - deliv)
            else:
                sent = sent + np.clip(ramp * np.clip(cmd, -CAP, CAP) - sent, -1.0, 1.0)
                deliv = deliv + alpha * (sent - deliv)
            dc.qpos[:7], dc.qvel[:7] = d.qpos[:7], 0.0
            mujoco.mj_forward(ctrl, dc)
            d.qfrc_applied[:7] = gain * deliv - fric * np.tanh(d.qvel[:7] / 0.05) + dc.qfrc_bias[:7]
            mujoco.mj_step(true, d)
            dqmax = np.maximum(dqmax, np.abs(d.qvel[:7]))
            if trip is None and (np.abs(d.qvel[:7]) > DQ_TRIP).any():
                trip = (k * 10 + _) / 1000
        z.append(1e3 * (d.site_xpos[site][2] - home[2]))
        xs.append(d.site_xpos[site] - home)
    run.xs = np.array(xs)
    return np.array(z), trip, dqmax, np.array(seen)


def real_z(path):
    z = np.load(path)
    p = z["episode_001_null_policy"]
    return 1e3 * p[1:, 4 + 6] / 5            # the handle height the policy saw, 10 ms old -> shift by one tick


def main():
    zr = real_z(sys.argv[1])
    show = range(2, 68, 3)
    print("real hand height (mm), every 30 ms from 0.03 s:   " + " ".join(f"{zr[i]:+4.0f}" for i in show))
    plants = []
    for arm, g, lag2, fr in itertools.product((0.1, 0.01), (1.0, 0.75, 0.6), (0.002, 0.015, 0.03), (0.0, 1.0)):
        gain = np.full(7, g)
        lag = np.full(7, 0.002)
        lag[1] = lag2
        plants.append((f"arm {arm:<4} gain {g:<4} lag j2 {1e3 * lag2:2.0f} ms fric {fr}",
                       dict(armature=arm, gain=gain, lag=lag, friction=fr * np.array([.5, 1.8, .4, .9, .1, .5, .1]))))
    for name, pl in plants:
        z, trip, dqmax, _ = run(pl)
        rms = np.sqrt(np.mean((z[:67] - zr[:67]) ** 2))
        print(f"{name:42s} trip {'%.2f s' % trip if trip else '  --  '} rms {rms:5.1f} | "
              + " ".join(f"{z[i]:+4.0f}" for i in show))


if __name__ == "__main__":
    main()
