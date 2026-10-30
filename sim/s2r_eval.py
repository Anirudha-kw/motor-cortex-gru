"""Held-out check before hardware: the numpy policy (policy_numpy.py) in a simulator built to differ from training
the way the real Panda does, run with the loop structure the robot will use.  CPU only (MuJoCo C), no training.

Plant (Menagerie panda_nohand.xml, plain MuJoCo, 1 ms steps, implicitfast integrator):
  gravity ON, compensated by the controller with a NOMINAL model (known tool included) while the true links' masses
  and inertias are perturbed: the compensation error is configuration-dependent, not a constant bias
  Coulomb friction through MuJoCo's constraint solver (dof_frictionloss), not the smoothed friction of training
  joint limits as hard constraints; contacts off (clear workspace)
  a real payload body (its weight is real and unknown to the compensation) and a known tool
Loop: the policy every 10 ms on the previous period's sensing (10 ms delay, as trained), its torque held between
updates; every 1 ms the command is capped (s2r.CAP), rate-limited to 1 N m per ms, passed through a 2 ms torque
lag and added to the gravity compensation; the curl field and constant push act at the handle every 1 ms.

Conditions: null, curl +35 N s/m, a 15 N sideways push, a 1.5 kg payload; plants: the nominal one, the friction
identified on the real Panda (Gaz et al. 2019, about 0.3-1.3 N m), and random draws from s2r's ranges.

    python s2r_eval.py out/meta_gru_broad.npz out/meta_gru_s2rR_s1.npz ...   -> out/s2r_eval.json
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from policy_numpy import GRUPolicy, Q_HOME

HERE = Path(__file__).resolve().parent
XML = Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
HANDLE = 0.207
CAP = np.array([40.0, 40.0, 40.0, 40.0, 8.0, 8.0, 8.0])
RATE = 1.0                      # N m per 1 ms step
LAG = 0.002                     # torque-loop time constant [s]
REACH, REACH_STEPS, GO = 0.12, 70, 10
CURL = 13.0 * 2.72
DQ_TRIP = np.array([1.85, 1.85, 1.85, 1.85, 2.2, 2.2, 2.2])            # the controller's joint-speed trip [rad/s]
GAZ_FRICTION = np.array([0.54, 0.87, 0.64, 1.27, 0.84, 0.30, 0.56])      # Gaz et al. 2019, phi_1 [N m]
CTRL_DAMPING = np.zeros(7)      # joint damping the deployment controller adds to the policy torque [N m s/rad]
DAMPING_SHAPE = np.array([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.1])            # per joint, x --ctrl-damping
# Plants fitted to the logged real reach (robot/tests/plant_fit.py, 2026-10-20): the policy in
# closed loop with these reproduces the real growing vertical oscillation (handle height rms 3.7-5.3 mm).  Torque
# delivery per joint: gain x first-order lag; Coulomb friction; armature.
REAL_FITS = [
    dict(armature=0.031, gain=[0.83, 0.62, 0.99, 1.02, 0.84, 0.96, 0.54], lag=[.018, .023, .037, .031, .015, .003, .029],
         friction=[0.47, 1.31, 0.12, 0.58, 0.3, 0.81, 0.01]),
    dict(armature=0.13, gain=[0.66, 0.82, 0.86, 1.02, 0.89, 0.73, 0.55], lag=[.003, .033, .039, .036, .004, .01, .008],
         friction=[1.2, 0.75, 1.94, 0.87, 0.52, 0.29, 0.01]),
    dict(armature=0.079, gain=[0.83, 0.65, 0.99, 0.98, 1.04, 0.58, 0.94], lag=[.015, .002, .01, .032, .004, .004, .019],
         friction=[0.71, 1.95, 1.4, 0.12, 0.05, 0.86, 0.37]),
    dict(armature=0.062, gain=[1.04, 0.86, 0.51, 0.77, 0.97, 0.91, 0.55], lag=[.025, .026, .005, .029, .007, .003, .011],
         friction=[0.98, 1.05, 0.79, 0.27, 0.15, 0.92, 0.47]),
]


def build(mass_scale=None, friction=None, damping=1.0, tool=0.0, payload=0.0, nominal=False, armature=None):
    spec = mujoco.MjSpec.from_file(str(XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):                   # keyframes carry the removed actuators' controls
        spec.delete(k)
    spec.option.timestep = 0.001
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    link7 = spec.body("link7")
    link7.add_site(name="handle", pos=[0, 0, HANDLE], size=[0.01] * 3)
    if tool > 0:
        b = link7.add_body(name="tool", pos=[0, 0, 0.157])
        b.mass, b.inertia, b.explicitinertial = tool, [1e-4] * 3, True
    if payload > 0 and not nominal:
        b = link7.add_body(name="payload", pos=[0, 0, HANDLE])
        b.mass, b.inertia, b.explicitinertial = payload, [0.4 * payload * 0.03 ** 2] * 3, True
    m = spec.compile()
    m.dof_damping[:7] = damping
    if armature is not None and not nominal:          # reflected motor inertia (Menagerie: 0.1 on every joint)
        m.dof_armature[:7] = armature
    if friction is not None and not nominal:
        m.dof_frictionloss[:7] = friction
    if mass_scale is not None and not nominal:
        for i in range(1, 8):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
            m.body_mass[bid] *= mass_scale[i - 1]
            m.body_inertia[bid] *= mass_scale[i - 1]
    return m


def handle_of(m, d, site):
    return d.site_xpos[site].copy()


def run(policy_path, plant, cond, rng, n_out=6, trace=None):
    """trace: a list to receive, per 10 ms tick, (sensing read now: handle, q, dq; target; go; commanded torque)."""
    delay = plant.get("delay", 1)                                           # controller periods of sensing delay
    true = build(plant["mass_scale"], plant["friction"], plant["damping"], plant["tool"], cond.get("payload", 0.0),
                 armature=plant.get("armature"))
    ctrl = build(tool=plant["tool"], nominal=True)                        # the controller's own model
    dt, dc = mujoco.MjData(true), mujoco.MjData(ctrl)
    site = mujoco.mj_name2id(true, mujoco.mjtObj.mjOBJ_SITE, "handle")
    site_c = mujoco.mj_name2id(ctrl, mujoco.mjtObj.mjOBJ_SITE, "handle")
    dt.qpos[:7] = Q_HOME
    mujoco.mj_forward(true, dt)
    dc.qpos[:7] = Q_HOME
    mujoco.mj_forward(ctrl, dc)
    home = dc.site_xpos[site_c].copy()
    pol = GRUPolicy(policy_path, home)
    angles = rng.permutation(8)[:n_out] * 2 * np.pi / 8
    targets = []
    for a in angles:
        targets += [home + REACH * np.array([np.cos(a), np.sin(a), 0.0]), home.copy()]
    tau_cmd, tau_applied, held = np.zeros(7), np.zeros(7), np.zeros(7)
    lag_j = np.asarray(plant.get("lag", np.full(7, LAG)), float)            # per-joint torque lag (fitted plants)
    gain_j = np.asarray(plant.get("gain", np.ones(7)), float)
    past = [(home.copy(), dt.qpos[:7].copy(), np.zeros(7))] * delay       # sensing in transit, oldest first
    jac = np.zeros((3, true.nv))
    rec = {"x": [], "v": [], "rate_sat": 0, "cap_sat": 0, "limit": 0, "tau": [], "dq_max": np.zeros(7)}
    for r, target in enumerate(targets):
        for s in range(REACH_STEPS):
            go = float(s >= GO + 1)
            handle, q, qd = past[0]
            nq = q + plant["noise"] * 1e-3 * rng.normal(size=7)
            nqd = qd + plant["noise"] * 0.01 * rng.normal(size=7)
            held = pol.step(target, go, handle, nq, nqd)
            # this period's sensing, read now, is what the policy sees next period (10 ms delay)
            dc.qpos[:7] = dt.qpos[:7]
            mujoco.mj_kinematics(ctrl, dc)
            past = past[1:] + [(dc.site_xpos[site_c].copy(), dt.qpos[:7].copy(), dt.qvel[:7].copy())]
            if trace is not None:
                trace.append((*past[-1], target.copy(), go, held.copy()))
            for _ in range(10):                                           # 1 kHz real-time loop
                target_cmd = np.clip(held - CTRL_DAMPING * dt.qvel[:7], -CAP, CAP)
                rec["cap_sat"] += int(np.any(np.abs(held) > CAP))
                step = np.clip(target_cmd - tau_cmd, -RATE, RATE)
                rec["rate_sat"] += int(np.any(np.abs(target_cmd - tau_cmd) > RATE))
                tau_cmd = tau_cmd + step
                tau_applied += (1 - np.exp(-0.001 / lag_j)) * (tau_cmd - tau_applied)
                dc.qpos[:7], dc.qvel[:7] = dt.qpos[:7], 0.0               # gravity compensation, nominal model
                mujoco.mj_forward(ctrl, dc)
                grav = dc.qfrc_bias[:7].copy()
                mujoco.mj_jacSite(true, dt, jac, None, site)
                v = jac @ dt.qvel
                f = np.zeros(3)
                if "curl" in cond:
                    f[:2] += cond["curl"] * np.array([v[1], -v[0]])
                if "push" in cond:
                    f += np.asarray(cond["push"])
                dt.qfrc_applied[:] = 0.0
                dt.qfrc_applied[:7] = gain_j * tau_applied + grav + (jac.T @ f)[:7]
                mujoco.mj_step(true, dt)
                lo, hi = true.jnt_range[:7, 0], true.jnt_range[:7, 1]
                rec["limit"] += int(np.any((dt.qpos[:7] < lo + 1e-3) | (dt.qpos[:7] > hi - 1e-3)))
                rec["dq_max"] = np.maximum(rec["dq_max"], np.abs(dt.qvel[:7]))
            mujoco.mj_jacSite(true, dt, jac, None, site)
            rec["x"].append(dt.site_xpos[site].copy())
            rec["v"].append((jac @ dt.qvel).copy())
            rec["tau"].append(tau_cmd.copy())
            if not np.all(np.isfinite(dt.qpos)) or np.linalg.norm(dt.site_xpos[site] - home) > 0.5:
                return dict(unstable=True)
    x, v = np.array(rec["x"]).reshape(len(targets), REACH_STEPS, 3), np.array(rec["v"]).reshape(len(targets), REACH_STEPS, 3)
    lateral, final, peak = [], [], []
    holds = [home] + targets[:-1]
    for r in range(0, len(targets), 2):                                     # outward reaches
        dvec = (targets[r] - holds[r])[:2]
        dvec /= np.linalg.norm(dvec)
        normal = np.array([-dvec[1], dvec[0]])
        speed = np.linalg.norm(v[r, :, :2], axis=1)
        k = int(speed.argmax())
        lateral.append(float(abs((x[r, k, :2] - holds[r][:2]) @ normal)))
        final.append(float(np.linalg.norm(x[r, -1] - targets[r])))
        peak.append(float(speed.max()))
    steps = len(targets) * REACH_STEPS * 10
    vertical = float(np.abs(x[..., 2] - home[2]).max())                   # reaches are horizontal: any height change
    return dict(unstable=False, lateral_mm=[1e3 * e for e in lateral], final_mm=[1e3 * e for e in final],
                peak_speed=peak, rate_sat=rec["rate_sat"] / steps, cap_sat=rec["cap_sat"] / steps,
                at_limit=rec["limit"] / steps, max_tau=np.abs(np.array(rec["tau"])).max(0).tolist(),
                vertical_mm=1e3 * vertical, dq_max=rec["dq_max"].tolist(),
                would_trip=bool((rec["dq_max"] > DQ_TRIP).any()))


def plants(rng, n_random=6):
    out = {"nominal": dict(mass_scale=np.ones(7), friction=np.zeros(7), damping=1.0, tool=0.0, noise=1.0),
           "real friction (Gaz 2019)": dict(mass_scale=np.ones(7), friction=GAZ_FRICTION, damping=1.0, tool=0.3,
                                            noise=1.0)}
    out["real friction, 20 ms delay"] = dict(out["real friction (Gaz 2019)"], delay=2)
    # the first hardware reach oscillated like a simulated arm with armature 0.01 (not the model's 0.1)
    out["real friction, armature 0.01"] = dict(out["real friction (Gaz 2019)"], armature=np.full(7, 0.01))
    out["real friction, armature 0.02"] = dict(out["real friction (Gaz 2019)"], armature=np.full(7, 0.02))
    out["real friction, armature 0.01, no tool"] = dict(out["real friction (Gaz 2019)"], armature=np.full(7, 0.01),
                                                        tool=0.0)
    out["real friction, armature 0.1, no tool"] = dict(out["real friction (Gaz 2019)"], tool=0.0)
    for arm in (0.01, 0.005):          # the 0.63 kg mass added on the robot (2026-10-20), declared or not
        out[f"real friction, armature {arm}, tool 0.63 kg"] = dict(out["real friction (Gaz 2019)"],
                                                                   armature=np.full(7, arm), tool=0.63)
    for t in (0.3, 0.6, 1.0):          # a known (gravity-compensated) mass at the flange restores inertia
        out[f"real friction, armature 0.01, tool {t} kg"] = dict(out["real friction (Gaz 2019)"],
                                                                 armature=np.full(7, 0.01), tool=t)
    for i, fit in enumerate(REAL_FITS):
        out[f"real fit {i}"] = dict(mass_scale=np.ones(7), friction=np.array(fit["friction"]), damping=0.0, tool=0.63,
                                    noise=1.0, armature=np.full(7, fit["armature"]), gain=np.array(fit["gain"]),
                                    lag=np.array(fit["lag"]))
    for i in range(n_random):
        out[f"random {i}"] = dict(mass_scale=rng.uniform(0.85, 1.15, 7),
                                  friction=rng.uniform(0, 1, 7) * np.array([1.5] * 4 + [1.0] * 3),
                                  damping=float(rng.uniform(0.5, 2.0)), tool=float(rng.uniform(0, 0.6)),
                                  noise=float(rng.uniform(0.5, 2.0)),
                                  armature=np.exp(rng.uniform(np.log(0.005), np.log(0.1), 7)))
    return out


CONDITIONS = {"null": {}, "curl +35": {"curl": CURL}, "push 15 N": {"push": (0.0, 15.0, 0.0)},
              "payload 1.5 kg": {"payload": 1.5}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+", type=Path)
    ap.add_argument("--random-plants", type=int, default=6)
    ap.add_argument("--caps", default="", help="torque caps, 7 comma values (default 40,40,40,40,8,8,8; the robot's "
                                               "first session uses 20,20,20,20,5,5,5)")
    ap.add_argument("--reach", type=float, default=0.06, help="reach radius [m]: the robot's 6 cm (sim probes: 0.12)")
    ap.add_argument("--only", default="", help="semicolon-separated plant names to run (default all)")
    ap.add_argument("--ctrl-damping", type=float, default=0.0,
                    help="joint damping added by the controller, N m s/rad on joints 1-4 (wrist x0.2, x0.1)")
    ap.add_argument("--conditions", default="", help="semicolon-separated condition names (default all)")
    ap.add_argument("--write", type=Path, default=HERE / "out" / "s2r_eval.json")
    args = ap.parse_args()
    global CAP, REACH, CTRL_DAMPING
    REACH = args.reach
    CTRL_DAMPING = args.ctrl_damping * DAMPING_SHAPE
    if args.caps:
        CAP = np.array([float(v) for v in args.caps.split(",")])
    res = {}
    for pp in args.policies:
        res[pp.name] = {}
        for pname, plant in plants(np.random.default_rng(7), args.random_plants).items():
            if args.only and pname not in args.only.split(";"):
                continue
            for cname, cond in CONDITIONS.items():
                if args.conditions and cname not in args.conditions.split(";"):
                    continue
                out = run(pp, plant, cond, np.random.default_rng(11))
                res[pp.name][f"{pname} | {cname}"] = out
                tag = "UNSTABLE" if out["unstable"] else (
                    f"lateral {np.round(out['lateral_mm'], 1)} mm  final {np.mean(out['final_mm']):.1f} mm  "
                    f"vertical max {out['vertical_mm']:.1f} mm  peak |dq| {max(out['dq_max']):.2f}"
                    f"{'  WOULD TRIP' if out['would_trip'] else ''}  capped {out['cap_sat']:.0%}")
                print(f"{pp.name:28s} {pname:24s} {cname:15s} {tag}", flush=True)
    args.write.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
