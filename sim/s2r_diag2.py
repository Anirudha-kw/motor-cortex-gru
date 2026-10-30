"""Are there rare gradient spikes, and which plant change makes them?  At fixed parameters (no update), the
gradient's global norm over many batches for: the original plant; the deployment limits at nominal dynamics (cap +
rate limit + controller state); the rate limit without the cap; and the full randomisation.  Quantiles show the
tail that a norm clip turns into a full-size step.  Run on a GPU node:

    python s2r_diag2.py --params out/meta_gru_broad.npz --batches 64
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import s2r
from meta import cost, rollout, train_episodes
from train import load_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--truncate", type=int, default=0)
    args = ap.parse_args()
    params = load_params(args.params)
    B = args.batch

    def make(dyn_fn, cap):
        def loss(params, key):
            k_ep, k_noise, k_dyn = jax.random.split(key, 3)
            episode = train_episodes(k_ep, B, "broad")
            return cost(rollout("gru", params, episode, k_noise, dyn=dyn_fn(k_dyn), cap=cap, truncate=args.truncate),
                        episode)[0]
        return jax.jit(jax.value_and_grad(loss))

    plants = {
        "original plant": make(lambda k: None, None),
        "limits, nominal dynamics": make(lambda k: s2r.nominal_dyn(B), s2r.CAP),
        "rate limit only (no cap)": make(lambda k: s2r.nominal_dyn(B), None),
        "full randomisation": make(lambda k: s2r.sample_dyn(k, B, 1.0), s2r.CAP),
    }
    for name, fn in plants.items():
        norms, losses = [], []
        for b in range(args.batches):
            val, g = fn(params, jax.random.PRNGKey(1000 + b))
            norms.append(float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))))
            losses.append(float(val))
        n = np.array(norms)
        print(f"{name:26s} loss {np.median(losses):.4f}  grad norm quantiles 50/90/99/max "
              f"{np.percentile(n, 50):.3g} / {np.percentile(n, 90):.3g} / {np.percentile(n, 99):.3g} / {n.max():.3g}  "
              f"batches above 1: {int((n > 1).sum())}/{len(n)}", flush=True)


if __name__ == "__main__":
    main()
