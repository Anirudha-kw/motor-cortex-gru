"""Train the GRU to drive the Panda's joint torques by backpropagating through MJX (run on a GPU node).

    python panda_train.py --iterations 3000 --batch 256

Writes <out>/panda_baseline.npz, a checkpoint every --save-every iterations, and a training log.
"""

import argparse
import time
from pathlib import Path

import jax
import optax

from panda import Env
from panda_cortex import center_out, init_params, loss_fn, random_reaches, reach_metrics, rollout
from train import save_params


def make_update(optimizer, batch):
    @jax.jit
    def update(params, opt_state, key):
        k_trials, k_noise = jax.random.split(key)
        trials = random_reaches(k_trials, batch)
        (_, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, trials, Env(), k_noise)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, parts
    return update


@jax.jit
def evaluate(params, key):
    trials = center_out()
    return reach_metrics(rollout(params, trials, Env(noise=0.0), key), trials)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"devices: {jax.devices()}  jax {jax.__version__}", flush=True)

    key = jax.random.PRNGKey(0)
    key, k_init = jax.random.split(key)
    params = init_params(k_init)
    schedule = optax.cosine_decay_schedule(args.lr, args.iterations, alpha=0.05)
    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = optimizer.init(params)
    update = make_update(optimizer, args.batch)

    t_last = time.perf_counter()
    for it in range(args.iterations + 1):
        key, k = jax.random.split(key)
        params, opt_state, parts = update(params, opt_state, k)
        if it % args.log_every == 0:
            m = evaluate(params, k)
            jax.block_until_ready(m)
            now = time.perf_counter()
            per_it = (now - t_last) / (args.log_every if it else 1)
            t_last = now
            print(f"{it:5d}  " + "  ".join(f"{name} {float(v):.4f}" for name, v in parts.items())
                  + f"  | centre-out: peak speed {float(m['peak_speed'].mean()):.2f} m/s, "
                  f"movement time {float(m['movement_time'].mean()) * 1e3:.0f} ms, "
                  f"endpoint error {float(m['endpoint_error'].mean()) * 1e3:.1f} mm, "
                  f"max lateral deviation {float(abs(m['max_perp']).mean()) * 1e3:.1f} mm"
                  f"  [{per_it * 1e3:.0f} ms/it]", flush=True)
        if it and it % args.save_every == 0:
            save_params(params, args.out / f"panda_ckpt_{it:05d}.npz")
    save_params(params, args.out / "panda_baseline.npz")
    print("saved", args.out / "panda_baseline.npz", flush=True)


if __name__ == "__main__":
    main()
