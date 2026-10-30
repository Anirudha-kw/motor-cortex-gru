"""The force-field adaptation experiment (as in adapt.py) on the GRU that drives the Panda.

Centre-out reaches in the horizontal plane: baseline, curl field (learning), washout.  Learning is
Adam on all of the network's weights with the training cost; after every update noise-free probes
measure lateral error at peak speed in the current field and the adaptation index in force-channel
trials.  Run on a GPU node:

    python panda_adapt.py --params out/panda_baseline.npz

Writes <out>/panda_adaptation.npz.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from panda import Env
from panda_cortex import CURL, center_out, loss_fn, reach_metrics, rollout
from train import load_params

NULL, FIELD, CHANNEL = Env(), Env(curl=CURL), Env(channel=1.0)


def adaptation_index(traj, trials):
    """Regression slope of the arm's lateral force in the channel onto the ideal curl-field compensation."""
    start, target = trials[1:3]
    direction = (target - start)[:, :2] / jnp.linalg.norm((target - start)[:, :2], axis=-1, keepdims=True)
    normal = jnp.stack([-direction[:, 1], direction[:, 0]], -1)
    v = traj["v"]
    field_force = CURL * jnp.stack([v[..., 1], -v[..., 0]], -1)
    ideal = -jnp.sum(field_force * normal, -1)
    produced = -jnp.sum(traj["channel_force"][..., :2] * normal, -1)
    return jnp.sum(produced * ideal) / jnp.sum(ideal * ideal)


@jax.jit
def probe(params, key):
    trials = center_out()
    quiet = lambda env: env._replace(noise=0.0)
    null = reach_metrics(rollout(params, trials, quiet(NULL), key), trials)
    field = reach_metrics(rollout(params, trials, quiet(FIELD), key), trials)
    channel = rollout(params, trials, quiet(CHANNEL), key)
    return {
        "perp_null": null["perp_at_peak_speed"].mean(),
        "perp_field": field["perp_at_peak_speed"].mean(),
        "endpoint_null": null["endpoint_error"].mean(),
        "endpoint_field": field["endpoint_error"].mean(),
        "adaptation_index": adaptation_index(channel, trials),
    }


@jax.jit
def snapshot(params, key):
    trials = center_out()
    return tuple(rollout(params, trials, env._replace(noise=0.0), key) for env in (NULL, FIELD, CHANNEL))


def make_update(optimizer, env, repeats):
    trials = center_out(repeats=repeats)

    @jax.jit
    def update(params, opt_state, key):
        (_, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, trials, env, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state
    return update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--updates-field", type=int, default=150)
    parser.add_argument("--updates-washout", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    params = load_params(args.params)
    key = jax.random.PRNGKey(7)
    optimizer = optax.adam(args.lr)
    curves = {k: [] for k in ("perp_null", "perp_field", "endpoint_null", "endpoint_field", "adaptation_index")}
    snaps = {}

    def record(params, key):
        for name, value in probe(params, key).items():
            curves[name].append(float(value))

    snaps["baseline"], snaps["early_field"], snaps["baseline_channel"] = snapshot(params, key)
    record(params, key)
    for phase, env, n in (("field", FIELD, args.updates_field), ("washout", NULL, args.updates_washout)):
        update = make_update(optimizer, env, args.repeats)
        opt_state = optimizer.init(params)
        for i in range(n):
            key, k = jax.random.split(key)
            params, opt_state = update(params, opt_state, k)
            record(params, k)
            if i % 25 == 0 or i == n - 1:
                print(f"{phase:8s} update {i + 1:3d}: lateral error at peak speed in field "
                      f"{curves['perp_field'][-1] * 1e3:+6.1f} mm, in null {curves['perp_null'][-1] * 1e3:+6.1f} mm, "
                      f"adaptation index {curves['adaptation_index'][-1]:.2f}, endpoint error field/null "
                      f"{curves['endpoint_field'][-1] * 1e3:.1f}/{curves['endpoint_null'][-1] * 1e3:.1f} mm", flush=True)
        null_traj, field_traj, channel_traj = snapshot(params, key)
        if phase == "field":
            snaps["late_field"], snaps["after_effect"], snaps["late_channel"] = field_traj, null_traj, channel_traj
        else:
            snaps["washed_out"] = null_traj

    trials = center_out()
    out = {f"curve_{k}": np.array(v) for k, v in curves.items()}
    out.update({"updates_field": args.updates_field, "updates_washout": args.updates_washout, "curl": CURL,
                "start": np.asarray(trials[1]), "target": np.asarray(trials[2]), "go": np.asarray(trials[3])})
    for name, traj in snaps.items():
        for k, v in traj.items():
            out[f"{name}_{k}"] = np.asarray(v)
    np.savez_compressed(args.out / "panda_adaptation.npz", **out)
    print("wrote", args.out / "panda_adaptation.npz", flush=True)


if __name__ == "__main__":
    main()
