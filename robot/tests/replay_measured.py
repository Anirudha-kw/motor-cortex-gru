"""Split the real-vs-sim gap in two with the 1 kHz log (motor_cortex_experiment.py --record, controller v2):

  1. link dynamics: drive the simulated arm with the joint torque the robot MEASURED (tau_J - gravity), from the
     recorded initial state, and compare its motion with the real one.  The torque sensors sit after the gearbox, so
     friction and rotor inertia are upstream of them: a correct rigid-body model needs neither.
  2. torque delivery: how the measured torque follows the commanded tau_J_d (lag, gain, friction), per joint.

    python3 replay_measured.py ~/mc_reach_logged.npz
"""
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

XML = Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
HANDLE = 0.207
Q, DQ, TJ, TD, SENT, G, EXT = (slice(2 + 7 * i, 9 + 7 * i) for i in range(7))


def model(armature, tool, damping=0.0):
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
    m.dof_frictionloss[:7] = 0.0
    m.dof_damping[:7] = damping
    return m


def uniform(f):
    """The running rows on a uniform 1 ms grid (the log drops a few rows)."""
    t = f[:, 0] - f[0, 0]
    tg = np.arange(0.0, t[-1], 0.001)
    return tg, np.stack([np.interp(tg, t, f[:, c]) for c in range(f.shape[1])], 1)


def replay(f, armature, tool, damping, torque):
    """Drive the sim with `torque` (n x 7, link-side, gravity excluded) from the first row's state."""
    m = model(armature, tool, damping)
    d, dg = mujoco.MjData(m), mujoco.MjData(m)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "handle")
    d.qpos[:7], d.qvel[:7] = f[0, Q], f[0, DQ]
    q, x = [], []
    for k in range(len(f)):
        dg.qpos[:7], dg.qvel[:7] = d.qpos[:7], 0.0
        mujoco.mj_forward(m, dg)
        d.qfrc_applied[:7] = torque[k] + dg.qfrc_bias[:7]
        mujoco.mj_step(m, d)
        q.append(d.qpos[:7].copy())
        x.append(d.site_xpos[site].copy())
    return np.array(q), np.array(x)


def main():
    z = np.load(sys.argv[1])
    key = [k for k in z.files if k.endswith("_fast") and k.startswith("episode_")][0]
    f = z[key]
    f = f[f[:, 1] == 2]
    t, f = uniform(f)
    window = t < float(sys.argv[2]) if len(sys.argv) > 2 else t < 0.40
    f = f[window]
    q_real = f[:, Q]
    measured = f[:, TJ] - f[:, G]
    commanded = f[:, TD]
    print(f"{key}: {len(f)} ms; real joint excursion over the window (mrad): "
          f"{np.round(1e3 * np.abs(q_real - q_real[0]).max(0), 0)}")
    print("\n1. sim driven by the MEASURED torque: rms joint error (mrad) per joint, and worst")
    res = []
    for arm, tool, damp in itertools.product((0.0, 0.01, 0.05, 0.1), (0.0, 0.63), (0.0, 1.0)):
        qs, _ = replay(f, arm, tool, damp, measured)
        e = 1e3 * np.sqrt(((qs - q_real) ** 2).mean(0))
        res.append((e.max(), arm, tool, damp, e))
    res.sort(key=lambda r: r[0])
    for r in res[:6]:
        print(f"   armature {r[1]:<5} tool {r[2]:<4} damping {r[3]:<4} worst {r[0]:6.1f}  per joint {np.round(r[4], 1)}")
    print("\n   same plants driven by the COMMANDED torque (tau_J_d) instead:")
    for r in res[:3]:
        qs, _ = replay(f, r[1], r[2], r[3], commanded)
        e = 1e3 * np.sqrt(((qs - q_real) ** 2).mean(0))
        print(f"   armature {r[1]:<5} tool {r[2]:<4} damping {r[3]:<4} worst {e.max():6.1f}  per joint {np.round(e, 1)}")


if __name__ == "__main__":
    main()
