"""Reference trace for test_core.cpp: a noise-free closed-loop run of the Python policy (s2r_eval.py, nominal plant,
curl field) recording, per 10 ms tick, the sensing the controller reads and the torque the policy commands.

    python robot/tests/make_trace.py weights.npz trace.txt
Line 1: home (3) and q at the start (7); then one line per tick: handle (3), q (7), dq (7), target (3), go, torque (7).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim"))
import s2r_eval  # noqa: E402

plant = dict(mass_scale=np.ones(7), friction=np.zeros(7), damping=1.0, tool=0.0, noise=0.0)
trace = []
out = s2r_eval.run(Path(sys.argv[1]), plant, {"curl": s2r_eval.CURL}, np.random.default_rng(11), n_out=3, trace=trace)
assert not out["unstable"], out
home = trace[0][3] if False else None
with open(sys.argv[2], "w") as f:
    from policy_numpy import Q_HOME
    import export_weights
    home = export_weights.home_position()
    f.write(" ".join(f"{v:.17g}" for v in [*home, *Q_HOME]) + "\n")
    for handle, q, dq, target, go, tau in trace:
        f.write(" ".join(f"{v:.17g}" for v in [*handle, *q, *dq, *target, go, *tau]) + "\n")
print(f"wrote {len(trace)} ticks to {sys.argv[2]}; lateral {np.round(out['lateral_mm'], 2)} mm")
