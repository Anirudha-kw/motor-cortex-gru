"""Train a policy across many random force fields so it can adapt within an episode (run on a GPU node).

    python meta_train.py --policy gru --fields meta --tag gru_meta
    python meta_train.py --policy mlp --fields meta --tag mlp_meta    # no memory: control
    python meta_train.py --policy gru --fields null --tag gru_null    # never sees a field: control
    python meta_train.py --policy gru --fields broad --tag gru_broad  # + constant forces and payloads
    python meta_train.py --policy gru --fields broad --tag gru_s2r --s2r --init out/meta_gru_broad.npz  # sim-to-real

Every --log-every iterations the frozen policy runs 6 centre-out-and-back reaches in each of three probe
environments (curl +35 N s/m, a constant 15 N sideways push, a 1.5 kg payload); in-context adaptation shows
up as the lateral error falling from the first outward reach to the last within one episode.
Writes <out>/meta_<tag>.npz and checkpoints.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import s2r
from panda import DT
from meta import CURL_UNIT, cost, init_policy, per_reach_metrics, rollout, test_episodes, train_episodes
from panda_cortex import CURL
from train import load_params, save_params


# The robot trips at 1.85 / 2.2 rad/s on its 6 cm reaches; training and the probes use 12 cm reaches, which ask
# about twice the joint speed for the same movement.  Both thresholds are scaled by that ratio, so the penalty
# catches oscillation (the fitted real plant without damping: 4.8 rad/s), not normal reaching (2.7 rad/s).
REACH_RATIO = 2.0
SPEED_TRIP = REACH_RATIO * jnp.array([1.85, 1.85, 1.85, 1.85, 2.2, 2.2, 2.2])
SPEED_SOFT = REACH_RATIO * jnp.array([1.5, 1.5, 1.5, 1.5, 1.9, 1.9, 1.9])


def speed_excess(q):
    """Mean squared joint speed above SPEED_SOFT, from the (T, B, nq) joint angles."""
    dq = jnp.diff(q[..., :7], axis=0) / DT
    return (jnp.clip(jnp.abs(dq) - SPEED_SOFT, 0) ** 2).sum(-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["gru", "mlp"], required=True)
    parser.add_argument("--fields", choices=["meta", "broad", "null"], default="meta")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hidden", type=int, default=128, help="GRU hidden units")
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--init", type=Path, default=None, help="warm start from these parameters")
    parser.add_argument("--s2r", action="store_true", help="sim-to-real plant (s2r.py): randomised dynamics, the "
                                                           "deployment controller's torque caps and rate limit")
    parser.add_argument("--s2r-ramp", type=int, default=1000, help="iterations over which the ranges grow 0 -> 1")
    parser.add_argument("--s2r-drop", default="", help="comma list of Dyn fields kept nominal, e.g. bias,viscous")
    parser.add_argument("--armature", default="", help="lo,hi: randomise each joint's armature log-uniformly "
                                                        "(kg m^2); empty keeps the model's 0.1")
    parser.add_argument("--nominal-limits", action="store_true",
                        help="control run: the s2r plant's caps, rate limit and buffer at nominal dynamics")
    parser.add_argument("--lag-joint", default="", help="lo,hi [s]: per-joint torque lag, log-uniform (s2r real)")
    parser.add_argument("--gain", default="", help="lo,hi: per-joint torque gain range (s2r real)")
    parser.add_argument("--friction-scale", type=float, default=0.0, help="x s2r.FRICTION_MAX (s2r real)")
    parser.add_argument("--ctrl-damping", default="", help="lo,hi: robot controller damping k per arm (s2r real)")
    parser.add_argument("--spike-cap", type=float, default=0.0,
                        help="never accept a gradient norm above this x the median of the first 50 iterations "
                             "(the rolling median alone drifted to 1e16 and let s2rA_s1 diverge)")
    parser.add_argument("--mask-unstable", type=float, default=0.0,
                        help="v_max [m/s]: arms whose handle ever moves faster (or goes non-finite) pass no gradient "
                             "(their backprop through an unstable closed loop explodes and swamps the batch: all three "
                             "s2rA runs diverged this way once the curriculum reached the full ranges); 0 = off")
    parser.add_argument("--speed-penalty", type=float, default=0.0,
                        help="weight on joint speed above SPEED_SOFT (the robot trips at 1.85/2.2 rad/s; a bounded "
                             "oscillation above it scored fine before); 0 = off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--truncate", type=int, default=0, help="stop the gradient through the arm's state every this "
                                                                "many steps (70 = every reach); 0: full horizon")
    parser.add_argument("--spike-factor", type=float, default=0.0,
                        help="skip an update whose gradient norm exceeds this x the median of the last 50 norms, "
                             "skipped ones included, so the threshold follows real drift (0: off)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"devices: {jax.devices()}  jax {jax.__version__}  {vars(args)}", flush=True)

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)
    params = init_policy(args.policy, k_init, args.hidden) if args.init is None else load_params(args.init)
    plant = args.s2r or args.nominal_limits
    drop = tuple(f for f in args.s2r_drop.split(",") if f)
    armature = tuple(float(v) for v in args.armature.split(",")) if args.armature else None
    cap = s2r.CAP if plant else None
    real = {}
    if args.lag_joint:
        real["lag"] = tuple(float(v) for v in args.lag_joint.split(","))
    if args.gain:
        real["gain"] = tuple(float(v) for v in args.gain.split(","))
    if args.friction_scale:
        real["friction_scale"] = args.friction_scale
    if args.ctrl_damping:
        real["ctrl_damping"] = tuple(float(v) for v in args.ctrl_damping.split(","))
    schedule = optax.cosine_decay_schedule(args.lr, args.iterations, alpha=0.05)
    # Backpropagating through 700 physics steps occasionally yields an infinite gradient; clipping would turn
    # it into NaN weights, so skip any update whose gradient is not finite (and count how often that happens).
    optimizer = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule)),
                                      max_consecutive_errors=100)
    opt_state = optimizer.init(params)

    def loss(params, key, scale):
        k_ep, k_noise, k_dyn = jax.random.split(key, 3)
        episode = train_episodes(k_ep, args.batch, args.fields)
        dyn = (s2r.sample_dyn(k_dyn, args.batch, scale, drop, armature, real) if args.s2r else
               s2r.nominal_dyn(args.batch) if args.nominal_limits else None)
        traj = rollout(args.policy, params, episode, k_noise, dyn=dyn, cap=cap, truncate=args.truncate)
        if args.mask_unstable <= 0:
            total, parts = cost(traj, episode)
            if args.speed_penalty > 0:
                parts["speed"] = args.speed_penalty * speed_excess(traj["q"])
                total = total + parts["speed"]
            return total, parts
        speed = jnp.linalg.norm(traj["v"], axis=-1)                                   # (T, B)
        ok = jax.lax.stop_gradient(jnp.all(jnp.isfinite(traj["x"]), axis=(0, 2))
                                   & (jnp.nan_to_num(speed, nan=jnp.inf).max(0) < args.mask_unstable))
        masked = {k: jnp.where(ok.reshape((1, -1) + (1,) * (v.ndim - 2)), v, jax.lax.stop_gradient(v))
                  for k, v in traj.items()}
        total, parts = cost(masked, episode)
        if args.speed_penalty > 0:
            parts["speed"] = args.speed_penalty * speed_excess(masked["q"])
            total = total + parts["speed"]
        return total, dict(parts, unstable=1.0 - ok.mean())

    @jax.jit
    def update(params, opt_state, key, scale, threshold):
        (_, parts), grads = jax.value_and_grad(loss, has_aux=True)(params, key, scale)
        norm = optax.global_norm(grads)
        new_updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, new_updates)
        ok = norm < threshold
        keep = lambda new, old: jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), new, old)
        return keep(new_params, params), keep(new_state, opt_state), parts, ~ok, norm

    probes = {
        "curl +35": test_episodes(jax.random.PRNGKey(1), 16, CURL * CURL_UNIT),
        "push 15 N": test_episodes(jax.random.PRNGKey(1), 16, jnp.zeros((2, 2)), (0.0, 15.0, 0.0)),
        "payload 1.5 kg": test_episodes(jax.random.PRNGKey(1), 16, jnp.zeros((2, 2)), payload=1.5),
    }

    @jax.jit
    def probe(params):
        out = {}
        for name, episode in probes.items():
            B = episode.go.shape[1]
            light = s2r.hard_dyn(B)._replace(armature=jnp.full((B, 7), 0.01))
            plants = [("", None), (" [s2r hard]", s2r.hard_dyn(B)), (" [armature 0.01]", light)]
            if real:
                plants += [(" [real fit 0, d0]", s2r.real_fit_dyn(B, 0, 0.0)), (" [real fit 0, d8]", s2r.real_fit_dyn(B, 0, 8.0)),
                           (" [real fit 1, d8]", s2r.real_fit_dyn(B, 1, 8.0))]
            for plant_name, dyn in plants:
                c = s2r.CAP if dyn is not None else None
                traj = rollout(args.policy, params, episode, jax.random.PRNGKey(2), 0.0, dyn, c)
                m = per_reach_metrics(traj, episode)
                dq = jnp.abs(jnp.diff(traj["q"][..., :7], axis=0)) / DT                       # (T-1, B, 7)
                out[name + plant_name] = (jnp.abs(m["perp_at_peak"][0::2]).mean(1), m["final_error"][0::2].mean(1),
                                          (dq > SPEED_TRIP).any((0, 2)).mean(), dq.max())
        return out

    t_last = time.perf_counter()
    recent, spikes, max_norm = [], 0, 0.0
    for it in range(args.iterations + 1):
        key, k = jax.random.split(key)
        scale = jnp.asarray(min(1.0, it / max(args.s2r_ramp, 1)), jnp.float32)
        threshold = (args.spike_factor * float(np.median(recent[-50:]))
                     if args.spike_factor > 0 and len(recent) >= 10 else float("inf"))
        if args.spike_cap > 0 and len(recent) >= 50:
            threshold = min(threshold, args.spike_cap * float(np.median(recent[:50])))
        params, opt_state, parts, spiked, norm = update(params, opt_state, k, scale, jnp.asarray(threshold))
        if np.isfinite(float(norm)):            # a non-finite gradient is skipped by apply_if_finite; keep it out
            recent.append(float(norm))          # of the reference so the threshold stays meaningful
        spikes += int(spiked)
        max_norm = max(max_norm, float(norm))
        if it % args.log_every == 0:
            results = probe(params)
            jax.block_until_ready(results)
            now = time.perf_counter()
            per_it = (now - t_last) / (args.log_every if it else 1)
            t_last = now
            print(f"{it:5d}  " + "  ".join(f"{name} {float(v):.4f}" for name, v in parts.items())
                  + f"  skipped updates {int(opt_state.total_notfinite)}  spikes skipped {spikes}  "
                  f"grad norm median {float(np.median(recent[-50:])):.3g} max {max_norm:.3g}  [{per_it * 1e3:.0f} ms/it]",
                  flush=True)
            max_norm = 0.0
            for name, (lateral, final, tripped, dq_max) in results.items():
                print(f"       {name:15s} |lateral error at peak speed| per outward reach: "
                      + " ".join(f"{float(e) * 1e3:5.1f}" for e in lateral)
                      + f" mm, final error {float(final.mean()) * 1e3:.1f} mm, would trip {float(tripped):.0%}, "
                      f"peak |dq| {float(dq_max):.2f}", flush=True)
        if it and it % args.save_every == 0:
            save_params(params, args.out / f"meta_{args.tag}_ckpt_{it:05d}.npz")
    save_params(params, args.out / f"meta_{args.tag}.npz")
    print("saved", args.out / f"meta_{args.tag}.npz", flush=True)


if __name__ == "__main__":
    main()
