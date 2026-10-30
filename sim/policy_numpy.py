"""The in-context GRU policy in plain numpy, for the robot PC (no JAX): one call per 10 ms controller period.

Inputs are exactly those of meta.rollout: the reach target and go cue, and the robot's own delayed sensing --
the handle position (from the joint angles through the arm's kinematics), the joint angles and velocities -- plus
an efference copy of the previous command.  The output is the joint-torque command in N m, on top of Franka's
gravity compensation, before the real-time loop's cap and rate limit (s2r.CAP, 1000 N m/s).

    pol = GRUPolicy("out/meta_gru_s2rR_s1.npz")
    pol.reset()
    tau = pol.step(target, go, handle, q, qd)      # every 10 ms; feed the PREVIOUS period's sensing (10 ms delay)

The handle is 0.207 m along link 7's z axis from link 7's origin (the flange is 0.107 m; a 10 cm handle beyond it).
"""
import numpy as np

Q_HOME = np.array([0.0, -0.35, 0.0, -2.3, 0.0, 1.95, 0.785])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GRUPolicy:
    def __init__(self, path, home):
        """path: an .npz of meta_train.py; home: the handle position at Q_HOME, in the same frame as `handle`."""
        p = np.load(path)
        self.W_x, self.b_x, self.W_h, self.b_h = p["W_x"], p["b_x"], p["W_h"], p["b_h"]
        self.W_out, self.b_out = p["W_out"], p["b_out"]
        self.home = np.asarray(home, float)
        self.reset()

    def reset(self):
        self.h = np.zeros(self.W_h.shape[0])
        self.u_prev = np.zeros(self.W_out.shape[1])

    def step(self, target, go, handle, q, qd):
        x = np.concatenate([(np.asarray(target) - self.home) * 5, [float(go)], (np.asarray(handle) - self.home) * 5,
                            np.asarray(q) - Q_HOME, np.asarray(qd) * 0.2, self.u_prev])
        rx, zx, nx = np.split(x @ self.W_x + self.b_x, 3)
        rh, zh, nh = np.split(self.h @ self.W_h + self.b_h, 3)
        r, z = _sigmoid(rx + rh), _sigmoid(zx + zh)
        n = np.tanh(nx + r * nh)
        self.h = (1 - z) * n + z * self.h
        u = np.tanh(self.h @ self.W_out + self.b_out)
        self.u_prev = u
        return u * TAU_MAX
