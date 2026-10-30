"""Export a trained GRU policy (meta_train.py .npz) for the robot's C++ controller (../robot).

Binary, little-endian: magic b"MCGRU001", int32 n_in, n_hidden, n_out, then float64 arrays W_x (n_in x 3H, row-major),
b_x (3H), W_h (H x 3H), b_h (3H), W_out (H x n_out), b_out (n_out), q_home (7), tau_max (7), home (3: the handle at
q_home in the training model, for a cross-check against the robot's own kinematics), then b"END".

    python export_weights.py out/meta_gru_s2rR_s1.npz ../robot/weights/motor_cortex_s2rR_s1.mcgru
"""
import argparse
import hashlib
import struct
from pathlib import Path

import mujoco
import numpy as np

from policy_numpy import Q_HOME, TAU_MAX

XML = Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
HANDLE = 0.207                                   # link7 origin -> flange 0.107 m, + a 10 cm handle, along link7 z


def home_position():
    """The handle at Q_HOME, as the training model (and s2r_eval.py) computes it."""
    spec = mujoco.MjSpec.from_file(str(XML))
    for k in list(spec.keys):
        spec.delete(k)
    spec.body("link7").add_site(name="handle", pos=[0, 0, HANDLE], size=[0.01] * 3)
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:7] = Q_HOME
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "handle")].copy()


def export(src, dst):
    p = np.load(src)
    n_in, three_h = p["W_x"].shape
    hidden, n_out = p["W_out"].shape
    assert three_h == 3 * hidden and p["W_h"].shape == (hidden, three_h) and n_out == 7
    home = home_position()
    with open(dst, "wb") as f:
        f.write(b"MCGRU001")
        f.write(struct.pack("<3i", n_in, hidden, n_out))
        for a in (p["W_x"], p["b_x"], p["W_h"], p["b_h"], p["W_out"], p["b_out"], Q_HOME, TAU_MAX, home):
            f.write(np.ascontiguousarray(a, dtype="<f8").tobytes())
        f.write(b"END")
    digest = hashlib.sha256(Path(dst).read_bytes()).hexdigest()
    print(f"wrote {dst}: n_in {n_in}, hidden {hidden}, n_out {n_out}; home {np.round(home, 4)}; sha256 {digest}")
    return home


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    export(args.src, args.dst)


if __name__ == "__main__":
    main()
