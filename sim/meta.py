"""In-context force-field adaptation on the Panda: multi-reach episodes in random force fields.

An episode is a sequence of reaches.  Each reach starts where the previous one should have ended, the
controller's internal state (GRU hidden state, adaptive estimates) is never reset, and the force field
is drawn once per episode (in training it may switch at a reach boundary).  A recurrent network trained
across many such episodes can only do well by inferring the field from what it senses and compensating
on later reaches: it adapts in its hidden state, with frozen weights.

Training families: 'meta' = velocity-dependent fields only; 'broad' = velocity fields, constant forces and
payloads (a mass held at the handle whose weight the robot's gravity compensation does not know about),
in random combinations; 'null' = none.

All controllers share one rollout through the MJX Panda (same 10 ms sensing delay, motor lag, noise and
torque limits):
  gru   recurrent policy; inputs include an efference copy of its previous torque command
  mlp   the same inputs through a feedforward network: no memory, so no adaptation across reaches
  osc   operational-space impedance control tracking a minimum-jerk path, with optional Slotine-Li
        adaptive compensation of a velocity-dependent field, a constant force and an unknown payload mass --
        the classical answer when the disturbance's structure is known.  Gains are arrays, so one
        compiled rollout serves every gain setting.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import panda
from cortex import N_HIDDEN, W_HIDDEN, W_HOLD, W_SMOOTH, gru_cell
from cortex import init_params as init_gru
from panda import DELAY, DT, N_JOINTS, NOISE_Q, NOISE_QD, Q_HI, Q_HOME, Q_LO, TAU_MAX
from panda_cortex import HOME, PULSE_MAX, REACH, W_EFFORT, W_LIMIT, W_POSTURE

REACH_STEPS = 70                 # 0.7 s per reach
N_REACHES_TRAIN = 10
GO_TRAIN = (5, 16)               # go cue 50-150 ms after the target appears
GO_TEST = 10
CATCH = 0.05
CURL_MAX = 50.0                  # training curl gains in [-50, 50] N s/m (the experiment used 35)
VISCOUS = (-25.0, 5.0)           # eigenvalues of the symmetric (viscous) part [N s/m]
P_NULL, P_SWITCH, P_PULSE = 0.2, 0.3, 0.15
CONST_MAX, PAYLOAD_MAX = 20.0, 2.0      # 'broad' family: constant forces up to 20 N, payloads up to 2 kg
P_CONST, P_PAYLOAD = 0.5, 0.5
GRAVITY = 9.81
N_OBS = 3 + 1 + 3 + N_JOINTS + N_JOINTS + N_JOINTS   # target, go, handle, q, qd, previous command
MLP_HIDDEN = 256
T_MINJERK = 0.28                 # classical reference duration: matches the GRU's reach speed [s]
CURL_UNIT = jnp.array([[0.0, 1.0], [-1.0, 0.0]])    # F = curl * (v_y, -v_x), as in the experiment


class Episode(NamedTuple):
    q0: jax.Array          # initial joint angles (B, 7)
    target: jax.Array      # current reach target, per step (T, B, 3)
    hold: jax.Array        # where the hand should wait before the go cue: the previous target (T, B, 3)
    go: jax.Array          # 1 once the go cue has been seen (T, B)
    t_go: jax.Array        # seconds since the go cue was seen, negative before (T, B)
    field: jax.Array       # velocity-dependent field matrix B, F = B v_xy (T, B, 2, 2)
    force: jax.Array       # constant force + perturbation pulses on the handle (T, B, 3)
    payload: jax.Array     # payload mass held at the handle [kg] (T, B)
    spring: jax.Array      # position-dependent field S, F = S (x - home)_xy (T, B, 2, 2); test only


# ----------------------------------------------------------------------------- episodes

def sample_fields(key, batch):
    """Curl plus a symmetric (viscous) part with eigenvalues in VISCOUS; P_NULL of them are null."""
    k = jax.random.split(key, 4)
    curl = jax.random.uniform(k[0], (batch,), minval=-CURL_MAX, maxval=CURL_MAX)
    phi = jax.random.uniform(k[1], (batch,), maxval=jnp.pi)
    eig = jax.random.uniform(k[2], (batch, 2), minval=VISCOUS[0], maxval=VISCOUS[1])
    rot = jnp.stack([jnp.stack([jnp.cos(phi), -jnp.sin(phi)], -1), jnp.stack([jnp.sin(phi), jnp.cos(phi)], -1)], -2)
    viscous = jnp.einsum("bij,bj,bkj->bik", rot, eig, rot)
    field = curl[:, None, None] * CURL_UNIT + viscous
    return jnp.where((jax.random.uniform(k[3], (batch,)) < P_NULL)[:, None, None], 0.0, field)


def _per_step(per_reach):
    """(R, B, ...) per-reach values -> (R * REACH_STEPS, B, ...) per-step values."""
    return jnp.repeat(per_reach, REACH_STEPS, axis=0)


def _assemble(q0, targets, holds, go, field, const, pulse=None, pulse_on=None, pulse_off=None, payload=None,
              spring=None):
    n_reaches, batch = go.shape
    payload = jnp.zeros((n_reaches, batch)) if payload is None else payload
    spring = jnp.zeros((n_reaches, batch, 2, 2)) if spring is None else spring
    s = jnp.tile(jnp.arange(REACH_STEPS), n_reaches)[:, None]                     # step within reach
    seen = _per_step(go) + DELAY
    force = _per_step(const)
    if pulse is not None:
        active = (s >= _per_step(pulse_on)) & (s < _per_step(pulse_off))
        force = force + _per_step(pulse) * active[..., None]
    return Episode(q0, _per_step(targets), _per_step(holds), (s >= seen).astype(jnp.float32),
                   (s - seen) * DT, _per_step(field), force, _per_step(payload), _per_step(spring))


def sample_constant(key, batch):
    """Constant forces up to CONST_MAX in a random 3-D direction, present with probability P_CONST."""
    k = jax.random.split(key, 3)
    direction = jax.random.normal(k[0], (batch, 3))
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    size = CONST_MAX * jax.random.uniform(k[1], (batch,)) * (jax.random.uniform(k[2], (batch,)) < P_CONST)
    return size[:, None] * direction


def sample_payload(key, batch):
    """Payloads of 0.2-PAYLOAD_MAX kg, present with probability P_PAYLOAD."""
    k = jax.random.split(key, 2)
    mass = jax.random.uniform(k[0], (batch,), minval=0.2, maxval=PAYLOAD_MAX)
    return mass * (jax.random.uniform(k[1], (batch,)) < P_PAYLOAD)


def train_episodes(key, batch, fields="meta", n_reaches=N_REACHES_TRAIN):
    """Chains of random reaches around the workspace.  fields='meta' draws a random velocity field per
    episode, 'broad' a random combination of velocity field, constant force and payload; either switches
    to a fresh draw at a random reach boundary with probability P_SWITCH.  'null' never applies any."""
    k = jax.random.split(key, 16)
    q0 = Q_HOME + jax.random.uniform(k[0], (batch, N_JOINTS), minval=-0.2, maxval=0.2)
    start = panda.handle_kinematics(q0)
    targets = HOME + jax.random.uniform(k[1], (n_reaches, batch, 3), minval=-1.0, maxval=1.0) * jnp.array(
        [0.16, 0.16, 0.08])
    holds = jnp.concatenate([start[None], targets[:-1]])
    go = jax.random.randint(k[2], (n_reaches, batch), *GO_TRAIN)
    go = jnp.where(jax.random.uniform(k[3], (n_reaches, batch)) < CATCH, 10 * REACH_STEPS, go)

    first, second = sample_fields(k[4], batch), sample_fields(k[5], batch)
    switch_at = jax.random.randint(k[6], (batch,), 2, n_reaches)
    switch = jax.random.uniform(k[7], (batch,)) < P_SWITCH
    after = (jnp.arange(n_reaches)[:, None] >= switch_at[None]) & switch[None]
    field = jnp.where(after[..., None, None], second[None], first[None])
    const = jnp.zeros((n_reaches, batch, 3))
    payload = jnp.zeros((n_reaches, batch))
    if fields == "broad":
        const = jnp.where(after[..., None], sample_constant(k[12], batch)[None], sample_constant(k[13], batch)[None])
        payload = jnp.where(after, sample_payload(k[14], batch)[None], sample_payload(k[15], batch)[None])
    if fields == "null":
        field = jnp.zeros_like(field)

    direction = jax.random.normal(k[8], (n_reaches, batch, 3))
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    size = PULSE_MAX * jax.random.uniform(k[9], (n_reaches, batch)) * (
        jax.random.uniform(k[10], (n_reaches, batch)) < P_PULSE)
    onset = go + jax.random.randint(k[11], (n_reaches, batch), 0, 40)
    offset = onset + jax.random.randint(jax.random.fold_in(k[11], 1), (n_reaches, batch), 5, 31)
    return _assemble(q0, targets, holds, go, field, const, size[..., None] * direction, onset, offset, payload)


def test_episodes(key, n_episodes, field, const=(0.0, 0.0, 0.0), n_out=6, off_after=None, payload=0.0,
                  spring=None):
    """Centre-out-and-back reaches (out, back, out, back, ...) from the home posture; outward reaches go
    to a fresh random order of the 8 directions in every episode.  The field (2x2), constant force,
    payload and spring (2x2) apply throughout, or only to reaches before index `off_after`."""
    n_reaches = 2 * n_out
    perm = jax.vmap(lambda k: jax.random.permutation(k, 8))(jax.random.split(key, n_episodes))[:, :n_out]
    angles = perm.T * 2 * jnp.pi / 8                                               # (n_out, E)
    out = HOME + REACH * jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)], -1)
    targets = jnp.stack([out, jnp.broadcast_to(HOME, out.shape)], 1).reshape(n_reaches, n_episodes, 3)
    holds = jnp.concatenate([jnp.broadcast_to(HOME, (1, n_episodes, 3)), targets[:-1]])
    on = jnp.ones(n_reaches) if off_after is None else (jnp.arange(n_reaches) < off_after).astype(jnp.float32)
    field = on[:, None, None, None] * jnp.broadcast_to(jnp.asarray(field), (n_reaches, n_episodes, 2, 2))
    const = on[:, None, None] * jnp.broadcast_to(jnp.asarray(const), (n_reaches, n_episodes, 3))
    payload = on[:, None] * jnp.full((n_reaches, n_episodes), payload)
    spring = on[:, None, None, None] * jnp.broadcast_to(jnp.zeros((2, 2)) if spring is None else jnp.asarray(spring),
                                                        (n_reaches, n_episodes, 2, 2))
    q0 = jnp.broadcast_to(Q_HOME, (n_episodes, N_JOINTS))
    return _assemble(q0, targets, holds, jnp.full((n_reaches, n_episodes), GO_TEST), field, const,
                     payload=payload, spring=spring)


# ----------------------------------------------------------------------------- controllers

def init_policy(kind, key, n_hidden=N_HIDDEN):
    if kind == "gru":
        params = init_gru(key, n_hidden, n_in=N_OBS, n_out=N_JOINTS, out_bias=0.0)
    elif kind == "mlp":
        k = jax.random.split(key, 3)
        dense = lambda k, n, m: jax.random.uniform(k, (n, m), minval=-1, maxval=1) / jnp.sqrt(n)
        params = {"W1": dense(k[0], N_OBS, MLP_HIDDEN), "b1": jnp.zeros(MLP_HIDDEN),
                  "W2": dense(k[1], MLP_HIDDEN, MLP_HIDDEN), "b2": jnp.zeros(MLP_HIDDEN),
                  "W_out": dense(k[2], MLP_HIDDEN, N_JOINTS), "b_out": jnp.zeros(N_JOINTS)}
    else:
        raise ValueError(kind)
    params["W_out"] = 0.1 * params["W_out"]          # start from a quiet arm
    return params


def osc_params(stiffness_rate=10.0, damping=40.0, gamma=0.0, gamma_const=0.0, gamma_mass=0.0):
    """Impedance: stiffness = damping * stiffness_rate [N/m].  Adaptation gains (Slotine-Li) for the
    velocity-field (gamma), constant-force (gamma_const) and payload-mass (gamma_mass) parameters."""
    return {"lam": jnp.asarray(stiffness_rate), "kd": jnp.asarray(damping),
            "gamma": jnp.concatenate([jnp.full(4, gamma), jnp.full(3, gamma_const), jnp.full(1, gamma_mass)])}


def _learned_act(kind, params, obs, h):
    if kind == "gru":
        h = gru_cell(params, obs, h)
        return jnp.tanh(h @ params["W_out"] + params["b_out"]), h
    z = jnp.tanh(obs @ params["W1"] + params["b1"])
    z = jnp.tanh(z @ params["W2"] + params["b2"])
    return jnp.tanh(z @ params["W_out"] + params["b_out"]), h


def _minimum_jerk(hold, target, t_go):
    tau = jnp.clip(t_go / T_MINJERK, 0.0, 1.0)[:, None]
    moving = ((t_go > 0) & (t_go < T_MINJERK)).astype(jnp.float32)[:, None]
    delta = target - hold
    x = hold + delta * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
    v = moving * delta * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T_MINJERK
    a = moving * delta * (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / T_MINJERK ** 2
    return x, v, a


def _osc_act(params, theta, handle, q, qd, J, M, target, hold, t_go):
    """Slotine-Li in task space: F = Lambda xdd_r - kd s - Y theta, theta' = gamma Y^T s, with the disturbance
    modelled as F = [B v_xy, 0] + c - m (xdd_r + g z): theta = (B11, B12, B21, B22, c_x, c_y, c_z, m)."""
    x_d, v_d, a_d = _minimum_jerk(hold, target, t_go)
    v = jnp.einsum("bij,bj->bi", J, qd)
    lam = params["lam"]
    v_r = v_d - lam * (handle - x_d)
    a_r = a_d - lam * (v - v_d)
    s = v - v_r
    M_inv = jnp.linalg.inv(M)
    Lam = jnp.linalg.inv(jnp.einsum("bij,bjk,blk->bil", J, M_inv, J))
    zeros, ones = jnp.zeros_like(v[:, 0]), jnp.ones_like(v[:, 0])
    Y = jnp.stack([jnp.stack([v[:, 0], v[:, 1], zeros, zeros, ones, zeros, zeros, -a_r[:, 0]], -1),
                   jnp.stack([zeros, zeros, v[:, 0], v[:, 1], zeros, ones, zeros, -a_r[:, 1]], -1),
                   jnp.stack([zeros, zeros, zeros, zeros, zeros, zeros, ones, -(a_r[:, 2] + GRAVITY)], -1)],
                  -2)                                                                 # F_disturbance = Y theta
    F = jnp.einsum("bij,bj->bi", Lam, a_r) - params["kd"] * s - jnp.einsum("bij,bj->bi", Y, theta)
    J_bar = jnp.einsum("bij,bkj,bkl->bil", M_inv, J, Lam)                           # dynamically consistent
    null = jnp.eye(N_JOINTS) - jnp.einsum("bji,bkj->bik", J, J_bar)
    posture = 20.0 * (Q_HOME - q) - 3.0 * qd
    tau = jnp.einsum("bji,bj->bi", J, F) + jnp.einsum("bij,bj->bi", null, posture)
    theta = jnp.clip(theta + DT * params["gamma"] * jnp.einsum("bji,bj->bi", Y, s), -150.0, 150.0)
    return jnp.clip(tau, -TAU_MAX, TAU_MAX) / TAU_MAX, theta


# ----------------------------------------------------------------------------- rollout, cost, metrics

def rollout(kind, params, episode, key, noise_scale=1.0, dyn=None, cap=None, truncate=0):
    """Simulate a batch of episodes.  Every output is time-major: (T, B, ...).  dyn (panda.Dyn) and cap: the
    sim-to-real plant (s2r.py); dyn=None is the nominal plant, unchanged.  truncate > 0: the gradient does not flow
    through the arm's physical state across every truncate-th step (it still flows through the controller's hidden
    state), which bounds the exploding gradients of long differentiable-physics horizons."""
    T, B = episode.go.shape
    buffer = DELAY + 1 if dyn is None else 3                         # s2r: per-arm delay of 1 or 2 steps

    @jax.checkpoint
    def tick(carry, xs):
        arm, ctrl, u_prev = carry
        target, hold, go, t_go, field, force, payload, spring, noise, cut = xs
        arm = jax.tree_util.tree_map(lambda a: jnp.where(cut, jax.lax.stop_gradient(a), a), arm)
        if dyn is None:
            sensed, ns = arm.sensed[0], noise_scale
        else:
            sensed = jnp.where((dyn.delay == 2)[:, None], arm.sensed[0], arm.sensed[1])
            ns = noise_scale * dyn.noise[:, None]
        handle = sensed[:, :3] + ns * NOISE_Q * 0.5 * noise[:, :3]
        q = sensed[:, 3:10] + ns * NOISE_Q * noise[:, 3:10]
        qd = sensed[:, 10:17] + ns * NOISE_QD * noise[:, 10:17]
        if kind == "osc":
            J, M = panda.model_terms(arm.d)
            u, ctrl = _osc_act(params, ctrl, handle, q, qd, J, M, target, hold, t_go)
        else:
            obs = jnp.concatenate([(target - HOME) * 5, go[:, None], (handle - HOME) * 5, q - Q_HOME, qd * 0.2,
                                   u_prev], -1)
            u, ctrl = _learned_act(kind, params, obs, ctrl)
        arm = panda.step_field(arm, u * TAU_MAX, field, force, noise_scale, noise[:, 17:], jnp.zeros_like(force),
                               payload, spring, HOME, dyn, cap)
        d = arm.d
        v = jnp.einsum("bij,bj->bi", jax.vmap(panda.handle_jacobian)(d), d.qvel)
        out = {"x": panda.handle_position(d), "v": v, "q": d.qpos, "u": u}
        if kind == "gru":
            out["h"] = ctrl
        return (arm, ctrl, u), out

    ctrl = jnp.zeros((B, 8)) if kind == "osc" else jnp.zeros((B, params["W_h"].shape[0] if kind == "gru" else 1))
    noise = jax.random.normal(key, (T, B, 17 + N_JOINTS))
    cut = (jnp.arange(T) % truncate == 0) & (jnp.arange(T) > 0) if truncate else jnp.zeros(T, bool)
    xs = (episode.target, episode.hold, episode.go, episode.t_go, episode.field, episode.force, episode.payload,
          episode.spring, noise, cut)
    _, traj = jax.lax.scan(tick, (panda.reset(episode.q0, buffer, dyn is not None), ctrl, jnp.zeros((B, N_JOINTS))), xs)
    return traj


def cost(traj, episode):
    desired = jnp.where(episode.go[..., None] > 0, episode.target, episode.hold)
    error = jnp.sqrt(jnp.sum((traj["x"] - desired) ** 2, -1) + 1e-8)
    q = traj["q"]
    parts = {
        "position": (error * (1 + W_HOLD * (1 - episode.go))).mean(),
        "effort": W_EFFORT * (traj["u"] ** 2).mean(),
        "posture": W_POSTURE * ((q - Q_HOME) ** 2).mean(),
        "limits": W_LIMIT * ((jnp.clip(Q_LO - q, 0) ** 2 + jnp.clip(q - Q_HI, 0) ** 2).sum(-1)).mean(),
    }
    if "h" in traj:
        parts["hidden"] = W_HIDDEN * (traj["h"] ** 2).mean()
        parts["smooth"] = W_SMOOTH * ((traj["h"][1:] - traj["h"][:-1]) ** 2).mean()
    return sum(parts.values()), parts


def per_reach_metrics(traj, episode):
    """(n_reaches, B) arrays for every reach: signed horizontal lateral error at peak speed, max |lateral
    error|, 3-D distance from the straight start-target line at peak speed and at its maximum (this one
    also sees a payload's sag), final distance to target, peak speed and effort (mean squared normalised
    torque)."""
    T, B = episode.go.shape
    R = T // REACH_STEPS
    split = lambda a: a.reshape(R, REACH_STEPS, *a.shape[1:])
    x, v, u = split(traj["x"]), split(traj["v"]), split(traj["u"])
    target, hold = split(episode.target)[:, 0], split(episode.hold)[:, 0]
    direction = (target - hold)[..., :2]
    direction = direction / (jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-9)
    normal = jnp.stack([-direction[..., 1], direction[..., 0]], -1)
    perp = jnp.sum((x[..., :2] - hold[:, None, :, :2]) * normal[:, None], -1)    # (R, S, B)
    speed = jnp.linalg.norm(v[..., :2], axis=-1)
    at_peak = jnp.take_along_axis(perp, speed.argmax(1)[:, None], 1)[:, 0]
    line = (target - hold) / (jnp.linalg.norm(target - hold, axis=-1, keepdims=True) + 1e-9)       # (R, B, 3)
    rel = x - hold[:, None]
    dev = jnp.linalg.norm(rel - jnp.sum(rel * line[:, None], -1, keepdims=True) * line[:, None], axis=-1)
    return {
        "perp_at_peak": at_peak,
        "max_perp": jnp.abs(perp).max(1),
        "dev_at_peak": jnp.take_along_axis(dev, speed.argmax(1)[:, None], 1)[:, 0],
        "max_dev": dev.max(1),
        "final_error": jnp.linalg.norm(x[:, -1] - target, axis=-1),
        "peak_speed": speed.max(1),
        "effort": (u ** 2).mean((1, 3)),
    }
