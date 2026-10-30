"""Open-loop torque replay: feed the torques the controller commanded on the real robot (telemetry, 100 Hz) into the
simulated Panda from the recorded initial state, through the same 1 N m/ms rate limit and Franka's 100 Hz torque
filter, and compare the hand's motion with the real one.  The policy is out of the loop, so any mismatch is the plant.
A grid over plant parameters (armature, friction scale, flange mass, torque gain, joint damping) finds what reproduces
the real trajectory over the first WINDOW seconds.

    python3 replay_fit.py ~/mc_reach_mass063.npz
"""
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
GAZ = np.array([0.54, 0.87, 0.64, 1.27, 0.84, 0.30, 0.56])
HANDLE = 0.207
WINDOW = 0.30                                   # s: before the real trajectory's large excursions


def model(armature, friction_scale, tool, damping):
    spec = mujoco.MjSpec.from_file(str(XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    spec.body("link7").add_site(name="handle", pos=[0, 0, HANDLE], size=[0.01] * 3)
    if tool > 0:
        b = spec.body("link7").add_body(name="tool", pos=[0, 0, 0.157])
        b.mass, b.inertia, b.explicitinertial = tool, [1e-4] * 3, True
    m = spec.compile()
    m.dof_armature[:7] = armature
    m.dof_frictionloss[:7] = GAZ * friction_scale
    m.dof_damping[:7] = damping
    return m


def replay(rows, armature, friction_scale, tool, damping, gain, steps):
    m = model(armature, friction_scale, tool, damping)
    d, dg = mujoco.MjData(m), mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "handle")
    d.qpos[:7], d.qvel[:7] = rows[0, 15:22], rows[0, 22:29]
    mujoco.mj_forward(m, d)
    sent, filt = np.zeros(7), np.zeros(7)
    alpha = 1 - np.exp(-2 * np.pi * 100 * 0.001)            # franka_hw low-pass, 100 Hz cutoff
    out = []
    for k in range(steps):
        cmd = rows[min(k, len(rows) - 1), 29:36]             # held for the 10 ms period
        for _ in range(10):
            sent = sent + np.clip(cmd - sent, -1.0, 1.0)      # rate limit against the last sent torque
            filt = filt + alpha * (sent - filt)
            dg.qpos[:7], dg.qvel[:7] = d.qpos[:7], 0.0
            mujoco.mj_forward(m, dg)                          # gravity compensation (the robot knows its load)
            d.qfrc_applied[:7] = gain * filt + dg.qfrc_bias[:7]
            mujoco.mj_step(m, d)
        out.append(d.site_xpos[site].copy())
    return np.array(out)


def main():
    z = np.load(sys.argv[1])
    a = z[[k for k in z.files if k.startswith("episode_") and not k.endswith(("_policy", "_fast"))][0]
          if any(k.startswith("episode_") for k in z.files) else z.files[0]]
    run = a[a[:, 1] == 2]
    n = int(WINDOW * 100)
    real = run[1:n + 1, 9:12]                                 # hand after each 10 ms period
    home = run[0, 9:12]
    print(f"real hand displacement at {WINDOW:.2f} s (mm): {np.round(1e3 * (real[-1] - home), 1)}")
    grid = dict(armature=(0.1, 0.03, 0.01), friction=(1.0, 0.5, 0.0), tool=(0.0, 0.63), damping=(1.0, 0.0),
                gain=(0.9, 1.0, 1.2, 1.5))
    res = []
    for arm, fr, tool, damp, gain in itertools.product(*grid.values()):
        sim = replay(run, arm, fr, tool, damp, gain, n)
        err = 1e3 * (sim - real)
        res.append((np.sqrt((err ** 2).mean()), arm, fr, tool, damp, gain, err[-1], 1e3 * (sim[-1] - home)))
    res.sort(key=lambda r: r[0])
    print("best plants (rms error over the window, mm; displacement at the end):")
    for r in res[:8]:
        print(f"  rms {r[0]:5.2f}  armature {r[1]:<5} friction x{r[2]:<4} tool {r[3]:<4} damping {r[4]:<4} gain {r[5]:<4} "
              f"end displacement {np.round(r[7], 1)}")
    base = [r for r in res if r[1] == 0.1 and r[2] == 1.0 and r[4] == 1.0 and r[5] == 1.0 and r[3] == 0.63][0]
    print(f"training-like plant (armature 0.1, friction x1, tool 0.63, damping 1, gain 1): rms {base[0]:.2f}, "
          f"end displacement {np.round(base[7], 1)}")


if __name__ == "__main__":
    main()
