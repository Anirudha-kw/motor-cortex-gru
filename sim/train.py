"""Train the GRU 'motor cortex' to make reaches in a null field, by backpropagating through the arm.

Writes out/cortex_baseline.npz.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from arm import Env
from cortex import center_out, init_params, loss_fn, random_reaches, reach_metrics, rollout

OUT = Path(__file__).resolve().parent / "out"
ITERATIONS = 4000
BATCH = 512


def save_params(params, path):
    np.savez(path, **{k: np.asarray(v) for k, v in params.items()})


def load_params(path):
    return {k: jnp.asarray(v) for k, v in np.load(path).items()}


def make_update(optimizer, batch, env):
    @jax.jit
    def update(params, opt_state, key):
        k_trials, k_noise = jax.random.split(key)
        trials = random_reaches(k_trials, batch)
        (loss, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, trials, env, k_noise)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, parts
    return update


@jax.jit
def evaluate(params, key):
    trials = center_out()
    return reach_metrics(rollout(params, trials, Env(noise=0.0), key), trials)


def main():
    key = jax.random.PRNGKey(0)
    key, k_init = jax.random.split(key)
    params = init_params(k_init)
    schedule = optax.cosine_decay_schedule(3e-3, ITERATIONS, alpha=0.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    update = make_update(optimizer, BATCH, Env())

    for it in range(ITERATIONS + 1):
        key, k = jax.random.split(key)
        params, opt_state, parts = update(params, opt_state, k)
        if it % 250 == 0:
            m = evaluate(params, k)
            print(f"{it:5d}  " + "  ".join(f"{name} {float(v):.4f}" for name, v in parts.items())
                  + f"  | centre-out: peak speed {float(m['peak_speed'].mean()):.2f} m/s, "
                  f"movement time {float(m['movement_time'].mean()) * 1e3:.0f} ms, "
                  f"endpoint error {float(m['endpoint_error'].mean()) * 1e3:.1f} mm, "
                  f"max lateral deviation {float(jnp.abs(m['max_perp']).mean()) * 1e3:.1f} mm", flush=True)

    OUT.mkdir(exist_ok=True)
    save_params(params, OUT / "cortex_baseline.npz")


if __name__ == "__main__":
    main()
