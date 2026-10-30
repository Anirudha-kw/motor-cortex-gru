"""The same GRU 'motor cortex', delayed-reach task and cost, driving the Panda's seven joint torques.

Differences from the human-arm version, all forced by the plant: the readout is tanh-scaled to the
Panda's torque limits instead of sigmoid muscle excitations; sensing is the robot's own (joint encoders
with one controller step of delay, and the handle position from its kinematics) instead of
proprioception and vision; a weak posture cost settles the 7-DoF arm's redundant null space; and the
curl field, channel and perturbation pulses are scaled by the 2.7x heavier endpoint mass.
"""

import jax
import jax.numpy as jnp

import panda
from cortex import CATCH, N_HIDDEN, W_HIDDEN, W_HOLD, W_SMOOTH, gru_cell
from cortex import init_params as init_gru
from panda import DELAY, DT, N_JOINTS, Q_HI, Q_HOME, Q_LO, TAU_MAX

T_TRIAL = 1.5
N_STEPS = round(T_TRIAL / DT)
N_OBS = 3 + 1 + 3 + N_JOINTS + N_JOINTS     # target, go cue, handle position, q, qd
W_EFFORT = 2.0
W_POSTURE = 0.01
W_LIMIT = 100.0
REACH = 0.12                                 # centre-out radius [m]
MASS_RATIO = 2.72                            # Panda / human horizontal endpoint mass
CURL = 13.0 * MASS_RATIO                     # [N s/m]
PULSE_MAX = 10.0 * MASS_RATIO                # [N]
PULSE_STEPS = (5, 31)                        # 50-300 ms
HOME = panda.handle_kinematics(Q_HOME[None])[0]


def init_params(key):
    params = init_gru(key, N_HIDDEN, n_in=N_OBS, n_out=N_JOINTS, out_bias=0.0)
    params["W_out"] = 0.1 * params["W_out"]     # start from a quiet arm: random +-40 N m torques make it flail
    return params


def policy(params, x, h):
    """Joint torque commands within the Panda's limits, and the new hidden state."""
    h = gru_cell(params, x, h)
    return jnp.tanh(h @ params["W_out"] + params["b_out"]) * TAU_MAX, h


def random_reaches(key, batch):
    """Random start postures and targets around the workspace, random go cues; half get a force pulse."""
    k = jax.random.split(key, 9)
    q0 = Q_HOME + jax.random.uniform(k[0], (batch, N_JOINTS), minval=-0.25, maxval=0.25)
    target = HOME + jax.random.uniform(k[1], (batch, 3), minval=-1.0, maxval=1.0) * jnp.array([0.16, 0.16, 0.08])
    go = jax.random.randint(k[2], (batch,), 10, 41)
    go = jnp.where(jax.random.uniform(k[3], (batch,)) < CATCH, 10 * N_STEPS, go)        # catch trials
    direction = jax.random.normal(k[4], (batch, 3))
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    size = PULSE_MAX * jax.random.uniform(k[5], (batch,)) * (jax.random.uniform(k[6], (batch,)) < 0.5)
    onset = go + jax.random.randint(k[7], (batch,), 0, 60)
    offset = onset + jax.random.randint(k[8], (batch,), *PULSE_STEPS)
    return q0, panda.handle_kinematics(q0), target, go, size[:, None] * direction, onset, offset


def center_out(n_targets=8, radius=REACH, go_step=20, repeats=1):
    """8 targets in the horizontal plane through the handle's home position."""
    angles = jnp.tile(jnp.arange(n_targets) * 2 * jnp.pi / n_targets, repeats)
    n = len(angles)
    q0 = jnp.broadcast_to(Q_HOME, (n, N_JOINTS))
    start = jnp.broadcast_to(HOME, (n, 3))
    target = start + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles), jnp.zeros(n)], -1)
    return q0, start, target, jnp.full((n,), go_step), jnp.zeros((n, 3)), jnp.full((n,), -1), jnp.full((n,), -1)


def rollout(params, trials, env, key):
    """Simulate one batch of trials.  Every output is time-major: (N_STEPS, B, ...)."""
    q0, start, target, go, pulse, onset, offset = trials

    @jax.checkpoint                              # recompute the physics on the backward pass: fits in 8 GB
    def tick(carry, inputs):
        state, h = carry
        t, noise = inputs
        target_seen = target * (t >= DELAY)
        go_seen = (t >= go + DELAY).astype(jnp.float32)[:, None]
        tau, h = policy(params, panda.observe(state, HOME, target_seen, go_seen, env, noise[:, :17]), h)
        pulsing = ((t >= onset) & (t < offset)).astype(jnp.float32)[:, None]
        state, wall = panda.step(state, tau, start, target, env, noise[:, 17:], pulse * pulsing)
        d = state.d
        out = {"x": panda.handle_position(d), "q": d.qpos, "qd": d.qvel, "u": tau / TAU_MAX, "h": h,
               "channel_force": wall,
               "v": jnp.einsum("bij,bj->bi", jax.vmap(panda.handle_jacobian)(d), d.qvel)}
        return (state, h), out

    noise = jax.random.normal(key, (N_STEPS, q0.shape[0], 17 + N_JOINTS))
    init = (panda.reset(q0), jnp.zeros((q0.shape[0], params["W_h"].shape[0])))
    _, traj = jax.lax.scan(tick, init, (jnp.arange(N_STEPS), noise))
    return traj


def cost(traj, trials):
    start, target, go = trials[1:4]
    after_go = (jnp.arange(N_STEPS)[:, None] >= go[None] + DELAY).astype(jnp.float32)
    desired = start[None] + after_go[..., None] * (target - start)[None]
    error = jnp.sqrt(jnp.sum((traj["x"] - desired) ** 2, -1) + 1e-8)
    q = traj["q"]
    parts = {
        "position": (error * (1 + W_HOLD * (1 - after_go))).mean(),
        "effort": W_EFFORT * (traj["u"] ** 2).mean(),
        "posture": W_POSTURE * ((q - Q_HOME) ** 2).mean(),
        "limits": W_LIMIT * ((jnp.clip(Q_LO - q, 0) ** 2 + jnp.clip(q - Q_HI, 0) ** 2).sum(-1)).mean(),
        "hidden": W_HIDDEN * (traj["h"] ** 2).mean(),
        "smooth": W_SMOOTH * ((traj["h"][1:] - traj["h"][:-1]) ** 2).mean(),
    }
    return sum(parts.values()), parts


def loss_fn(params, trials, env, key):
    return cost(rollout(params, trials, env, key), trials)


def reach_metrics(traj, trials):
    """Like cortex.reach_metrics, with lateral error measured in the horizontal plane."""
    start, target = trials[1:3]
    speed = jnp.linalg.norm(traj["v"], axis=-1)
    peak = speed.max(0)
    moving = speed > 0.05 * peak[None]
    t = jnp.arange(N_STEPS)[:, None]
    onset = jnp.where(moving, t, N_STEPS).min(0)
    offset = jnp.where(moving, t, -1).max(0)
    direction = (target - start)[:, :2] / jnp.linalg.norm((target - start)[:, :2], axis=-1, keepdims=True)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
    perp = jnp.sum((traj["x"][..., :2] - start[None, :, :2]) * normal[None], -1)
    idx = jnp.abs(perp).argmax(0)
    return {
        "peak_speed": peak,
        "movement_time": (offset - onset) * DT,
        "onset": onset,
        "endpoint_error": jnp.linalg.norm(traj["x"][-1] - target, axis=-1),
        "max_perp": jnp.take_along_axis(perp, idx[None], 0)[0],
        "perp_at_peak_speed": jnp.take_along_axis(perp, speed.argmax(0)[None], 0)[0],
    }
