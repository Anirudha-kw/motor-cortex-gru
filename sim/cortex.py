"""GRU 'motor cortex' that drives the arm's muscles, the delayed-reach task, and the optimisation cost.

Each trial: the target appears at t = 0, a go cue follows 100-400 ms later, and the hand must stay put
until the cue and then reach the target.  Both cues are visual, so the network sees them 70 ms late;
the cost only asks for the target once the cue can have been seen, holding still before that is
weighted more heavily, and 10% of trials are catch trials with no go cue, as in real delayed-reach
tasks -- so anticipating the cue never pays.
The cost is the classic optimal-control trade-off -- task error plus muscle effort -- plus penalties
on hidden-state size and rate of change that favour smooth, low-energy population dynamics
(Sussillo et al. 2015).  Straight hand paths and bell-shaped speed profiles are not imposed: if they
appear, they emerge from this optimisation.
"""

import jax
import jax.numpy as jnp

import arm
from arm import DELAY_VISION, DT, N_OBS, Q_CENTER, Q_HI, Q_LO, hand_position, hand_velocity

T_TRIAL = 1.5
N_STEPS = round(T_TRIAL / DT)
N_HIDDEN = 128
W_EFFORT = 24.0
W_LIMIT = 100.0
W_HOLD = 4.0         # extra weight on position error before the go cue is seen
CATCH = 0.1
PULSE_MAX = 10.0     # [N]
PULSE_STEPS = (5, 31)  # 50-300 ms
W_HIDDEN = 1e-3
W_SMOOTH = 0.1


def init_params(key, n_hidden=N_HIDDEN, n_in=N_OBS, n_out=6, out_bias=-2.0):
    """GRU + linear readout, initialised like torch.nn.GRUCell / nn.Linear (out_bias -2: muscles start quiet)."""
    ks = jax.random.split(key, 5)
    bound = 1 / jnp.sqrt(n_hidden)
    uniform = lambda k, shape: jax.random.uniform(k, shape, minval=-bound, maxval=bound)
    return {
        "W_x": uniform(ks[0], (n_in, 3 * n_hidden)),        # input weights for (reset, update, candidate)
        "W_h": uniform(ks[1], (n_hidden, 3 * n_hidden)),    # recurrent weights
        "b_x": uniform(ks[2], (3 * n_hidden,)),
        "b_h": uniform(ks[3], (3 * n_hidden,)),
        "W_out": uniform(ks[4], (n_hidden, n_out)),
        "b_out": jnp.full((n_out,), out_bias),
    }


def gru_cell(params, x, h):
    gx = x @ params["W_x"] + params["b_x"]
    gh = h @ params["W_h"] + params["b_h"]
    rx, zx, nx = jnp.split(gx, 3, -1)
    rh, zh, nh = jnp.split(gh, 3, -1)
    r, z = jax.nn.sigmoid(rx + rh), jax.nn.sigmoid(zx + zh)
    n = jnp.tanh(nx + r * nh)
    return (1 - z) * n + z * h


def gru(params, x, h):
    """Muscle excitations in (0, 1) and the new hidden state."""
    h = gru_cell(params, x, h)
    return jax.nn.sigmoid(h @ params["W_out"] + params["b_out"]), h


def random_reaches(key, batch):
    """Random start postures, random targets anywhere in the workspace, random go-cue times.

    Half of the reaches also get a force pulse on the hand (up to 10 N for 50-300 ms, random direction
    and timing), so the network learns to correct with feedback rather than rely on feedforward alone.
    It never experiences a velocity-dependent field.
    """
    k = jax.random.split(key, 9)
    lo, hi = jnp.array([0.2, 0.9]), jnp.array([1.4, 2.2])
    q0 = lo + (hi - lo) * jax.random.uniform(k[0], (batch, 2))
    target = hand_position(lo + (hi - lo) * jax.random.uniform(k[1], (batch, 2)))
    go = jax.random.randint(k[2], (batch,), 10, 41)
    go = jnp.where(jax.random.uniform(k[3], (batch,)) < CATCH, 10 * N_STEPS, go)     # catch trials: no go cue
    angle = jax.random.uniform(k[4], (batch,), maxval=2 * jnp.pi)
    size = PULSE_MAX * jax.random.uniform(k[5], (batch,)) * (jax.random.uniform(k[6], (batch,)) < 0.5)
    pulse = size[:, None] * jnp.stack([jnp.cos(angle), jnp.sin(angle)], -1)
    onset = go + jax.random.randint(k[7], (batch,), 0, 60)
    offset = onset + jax.random.randint(k[8], (batch,), *PULSE_STEPS)
    return q0, hand_position(q0), target, go, pulse, onset, offset


def center_out(n_targets=8, radius=0.10, go_step=20, repeats=1):
    """The standard experiment: 8 targets 10 cm from a central start posture."""
    angles = jnp.tile(jnp.arange(n_targets) * 2 * jnp.pi / n_targets, repeats)
    q0 = jnp.broadcast_to(Q_CENTER, (len(angles), 2))
    start = hand_position(q0)
    target = start + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1)
    n = len(angles)
    return q0, start, target, jnp.full((n,), go_step), jnp.zeros((n, 2)), jnp.full((n,), -1), jnp.full((n,), -1)


def rollout(params, trials, env, key):
    """Simulate one batch of trials.  Every output is time-major: (N_STEPS, B, ...)."""
    q0, start, target, go, pulse, onset, offset = trials

    def tick(carry, inputs):
        state, h = carry
        t, noise = inputs
        target_seen = target * (t >= DELAY_VISION)
        go_seen = (t >= go + DELAY_VISION).astype(jnp.float32)[:, None]
        u, h = gru(params, arm.observe(state, target_seen, go_seen, env, noise[:, :14]), h)
        pulsing = ((t >= onset) & (t < offset)).astype(jnp.float32)[:, None]
        state, wall = arm.step(state, u, start, target, env, noise[:, 14:], pulse * pulsing)
        out = {"x": hand_position(state.q), "v": hand_velocity(state.q, state.qd), "q": state.q, "u": u,
               "act": state.act, "h": h, "channel_force": wall}
        return (state, h), out

    noise = jax.random.normal(key, (N_STEPS, q0.shape[0], 20))     # all sensory + motor noise, drawn at once
    init = (arm.reset(q0), jnp.zeros((q0.shape[0], params["W_h"].shape[0])))
    _, traj = jax.lax.scan(tick, init, (jnp.arange(N_STEPS), noise), unroll=2)
    return traj


def cost(traj, trials):
    start, target, go = trials[1:4]
    after_go = (jnp.arange(N_STEPS)[:, None] >= go[None] + DELAY_VISION).astype(jnp.float32)
    desired = start[None] + after_go[..., None] * (target - start)[None]
    error = jnp.sqrt(jnp.sum((traj["x"] - desired) ** 2, -1) + 1e-8)     # Euclidean: no preferred direction
    parts = {
        "position": (error * (1 + W_HOLD * (1 - after_go))).mean(),
        "effort": W_EFFORT * (traj["u"] ** 2).mean(),
        "hidden": W_HIDDEN * (traj["h"] ** 2).mean(),
        "smooth": W_SMOOTH * ((traj["h"][1:] - traj["h"][:-1]) ** 2).mean(),
        # The soft joint limits push back passively; without this term the optimiser learns to bounce
        # off them as free propulsion when effort is expensive.
        "limits": W_LIMIT * ((jnp.clip(Q_LO - traj["q"], 0) ** 2 + jnp.clip(traj["q"] - Q_HI, 0) ** 2).sum(-1)).mean(),
    }
    return sum(parts.values()), parts


def loss_fn(params, trials, env, key):
    total, parts = cost(rollout(params, trials, env, key), trials)
    return total, parts


def reach_metrics(traj, trials):
    """Per-reach peak speed, movement time, endpoint error, and signed max perpendicular deviation."""
    start, target = trials[1:3]
    speed = jnp.linalg.norm(traj["v"], axis=-1)                  # (T, B)
    peak = speed.max(0)
    moving = speed > 0.05 * peak[None]
    t = jnp.arange(N_STEPS)[:, None]
    onset = jnp.where(moving, t, N_STEPS).min(0)
    offset = jnp.where(moving, t, -1).max(0)
    direction = (target - start) / jnp.linalg.norm(target - start, axis=-1, keepdims=True)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
    perp = jnp.sum((traj["x"] - start[None]) * normal[None], -1)
    idx = jnp.abs(perp).argmax(0)
    at_peak = jnp.take_along_axis(perp, speed.argmax(0)[None], 0)[0]
    return {
        "peak_speed": peak,
        "movement_time": (offset - onset) * DT,
        "onset": onset,
        "endpoint_error": jnp.linalg.norm(traj["x"][-1] - target, axis=-1),
        "max_perp": jnp.take_along_axis(perp, idx[None], 0)[0],
        "perp_at_peak_speed": at_peak,
    }
