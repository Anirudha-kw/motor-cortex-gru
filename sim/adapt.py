"""Force-field adaptation experiment (Shadmehr & Mussa-Ivaldi 1994) on the trained GRU 'motor cortex'.

Protocol, all with centre-out reaches to 8 targets:
  1. baseline in the null field;
  2. curl field switched on -- first exposure with the unchanged network;
  3. adaptation: the network keeps learning (gradient descent on its weights) in the field;
  4. washout: the field is switched off -- after-effects, then de-adaptation.

Learning is gradient descent (Adam) on all of the network's weights, with the training cost.  After
every update, probe reaches measure (a) hand-path error in the current field, (b) the lateral force the
arm exerts in a force channel (Scheidt et al. 2000), regressed onto the force that would exactly cancel
the curl field -- the adaptation index (0 = no compensation, 1 = full) -- and (c) co-contraction, the
activity shared by each antagonist muscle pair, which stiffens the arm (Franklin et al. 2003).

Writes out/adaptation.npz.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from arm import CURL, Env
from cortex import center_out, loss_fn, reach_metrics, rollout
from train import OUT, load_params

UPDATES_FIELD = 150       # long enough for the kinematic error to reach zero, as in a human session
UPDATES_WASHOUT = 150
LEARNING_RATE = 1e-4      # larger steps (or plain SGD) make the learning curves oscillate
REPEATS = 16              # centre-out batch per update: 8 targets x 16 repeats (with sensory and motor noise)

NULL, FIELD, CHANNEL = Env(), Env(curl=CURL), Env(channel=1.0)


def adaptation_index(traj, trials):
    """Regression slope of the arm's lateral force in the channel onto the ideal curl-field compensation."""
    start, target = trials[1:3]
    direction = (target - start) / jnp.linalg.norm(target - start, axis=-1, keepdims=True)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
    v = traj["v"]
    field_force = CURL * jnp.stack([v[..., 1], -v[..., 0]], -1)          # what the curl field would push
    ideal = -jnp.sum(field_force * normal, -1)                            # lateral force that cancels it
    produced = -jnp.sum(traj["channel_force"] * normal, -1)               # arm's push against the channel wall
    return jnp.sum(produced * ideal) / jnp.sum(ideal * ideal)


def cocontraction(traj):
    """Mean activation shared by the three antagonist pairs (flexor/extensor), over the whole trial."""
    act = traj["act"]
    return jnp.minimum(act[..., 0::2], act[..., 1::2]).sum(-1).mean()


@jax.jit
def probe(params, key):
    """Noise-free centre-out reaches in the null field, in the curl field, and in a force channel."""
    trials = center_out()
    quiet = lambda env: env._replace(noise=0.0)
    null = rollout(params, trials, quiet(NULL), key)
    field = rollout(params, trials, quiet(FIELD), key)
    channel = rollout(params, trials, quiet(CHANNEL), key)
    return {
        "perp_null": reach_metrics(null, trials)["perp_at_peak_speed"].mean(),
        "perp_field": reach_metrics(field, trials)["perp_at_peak_speed"].mean(),
        "adaptation_index": adaptation_index(channel, trials),
        "cocontraction_null": cocontraction(null),
        "cocontraction_field": cocontraction(field),
    }


def make_update(optimizer, env):
    trials = center_out(repeats=REPEATS)

    @jax.jit
    def update(params, opt_state, key):
        (_, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, trials, env, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state
    return update


@jax.jit
def snapshot(params, key):
    """Full noise-free trajectories of the 8 centre-out reaches in the null field, curl field and channel."""
    trials = center_out()
    return tuple(rollout(params, trials, env._replace(noise=0.0), key) for env in (NULL, FIELD, CHANNEL))


def main():
    params = load_params(OUT / "cortex_baseline.npz")
    key = jax.random.PRNGKey(7)
    optimizer = optax.adam(LEARNING_RATE)
    curves = {k: [] for k in ("perp_null", "perp_field", "adaptation_index", "cocontraction_null",
                              "cocontraction_field")}
    snaps = {}

    def record(params, key):
        for name, value in probe(params, key).items():
            curves[name].append(float(value))

    snaps["baseline"], snaps["early_field"], snaps["baseline_channel"] = snapshot(params, key)
    record(params, key)

    for phase, env, n in (("field", FIELD, UPDATES_FIELD), ("washout", NULL, UPDATES_WASHOUT)):
        update = make_update(optimizer, env)
        opt_state = optimizer.init(params)
        for i in range(n):
            key, k = jax.random.split(key)
            params, opt_state = update(params, opt_state, k)
            record(params, k)
            if i % 25 == 0 or i == n - 1:
                print(f"{phase:8s} update {i + 1:3d}: lateral error at peak speed in field {curves['perp_field'][-1] * 1e3:+6.1f} mm,"
                      f" in null {curves['perp_null'][-1] * 1e3:+6.1f} mm, adaptation index "
                      f"{curves['adaptation_index'][-1]:.2f}, co-contraction (field) "
                      f"{curves['cocontraction_field'][-1]:.3f}", flush=True)
        null_traj, field_traj, channel_traj = snapshot(params, key)
        if phase == "field":
            snaps["late_field"], snaps["after_effect"], snaps["late_channel"] = field_traj, null_traj, channel_traj
        else:
            snaps["washed_out"] = null_traj

    trials = center_out()
    out = {f"curve_{k}": np.array(v) for k, v in curves.items()}
    out.update({"updates_field": UPDATES_FIELD, "updates_washout": UPDATES_WASHOUT,
                "start": np.asarray(trials[1]), "target": np.asarray(trials[2]), "go": np.asarray(trials[3])})
    for name, traj in snaps.items():
        for k, v in traj.items():
            out[f"{name}_{k}"] = np.asarray(v)
    np.savez_compressed(OUT / "adaptation.npz", **out)
    print("wrote", OUT / "adaptation.npz")


if __name__ == "__main__":
    main()
