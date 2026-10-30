"""Which part of the sim-to-real plant breaks the gradient?  At fixed parameters (no update), the loss and the
gradient's global norm and largest entry for the nominal plant and for each randomised component alone, at the
ramp's early scale and at full scale, over several batches.  Run on a GPU node:

    python s2r_diag.py --params out/meta_gru_broad.npz
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import s2r
from meta import cost, rollout, train_episodes
from train import load_params

FIELDS = ("friction", "viscous", "bias", "gain", "mass_scale", "tool", "lag", "delay", "noise")


def only(full, nominal, field):
    return nominal._replace(**{field: getattr(full, field)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=4)
    args = ap.parse_args()
    params = load_params(args.params)

    def loss(params, key, dyn):
        k_ep, k_noise = jax.random.split(key)
        episode = train_episodes(k_ep, args.batch, "broad")
        return cost(rollout("gru", params, episode, k_noise, dyn=dyn, cap=s2r.CAP), episode)[0]

    grad = jax.jit(jax.value_and_grad(loss))
    nominal = s2r.nominal_dyn(args.batch)
    for scale in (0.1, 1.0):
        rows = {"nominal": []} | {f: [] for f in FIELDS} | {"all": []}
        for b in range(args.batches):
            key = jax.random.PRNGKey(100 + b)
            full = s2r.sample_dyn(jax.random.fold_in(key, 7), args.batch, scale)
            for name in rows:
                dyn = nominal if name == "nominal" else full if name == "all" else only(full, nominal, name)
                val, g = grad(params, key, dyn)
                leaves = jax.tree_util.tree_leaves(g)
                norm = float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves)))
                biggest = float(max(jnp.max(jnp.abs(x)) for x in leaves))
                rows[name].append((float(val), norm, biggest))
        print(f"scale {scale}: loss / gradient global norm / largest entry, median over {args.batches} batches (max norm)")
        for name, r in rows.items():
            r = np.array(r)
            print(f"  {name:11s} loss {np.median(r[:, 0]):.4f}  norm {np.median(r[:, 1]):10.3g} (max {r[:, 1].max():10.3g})  "
                  f"largest {np.median(r[:, 2]):10.3g}", flush=True)


if __name__ == "__main__":
    main()
