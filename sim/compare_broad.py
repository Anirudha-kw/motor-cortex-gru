"""Does training on constant forces and payloads fix the failure case?  Frozen controllers on the Panda.

Controllers (identical episodes, sensor/motor noise and limits):
  GRU (broad-trained)      meta_gru_broad.npz: velocity fields + constant forces + payloads
  GRU (velocity fields)    meta_gru_meta.npz: the previous meta-trained GRU
  MLP (broad-trained)      meta_mlp_broad.npz: same training as the broad GRU, no memory
  Impedance, compliant / stiff
  Adaptive (Slotine-Li)    impedance + adaptive estimates of a velocity field, a constant force and the
                           payload mass (the textbook use of adaptive control); each gain tuned in turn on
                           separate episodes of its own disturbance
Conditions:
  inside the broad family: null; curl +35 / -35; mixed; constant 15 N push; payload 1.5 kg; payload 1.5 kg
  with curl +35
  beyond it: curl +70 (twice the test field, 1.4x the trained maximum of 50); a spring pulling 300 N/m toward a
  point 5 cm to the left (position-dependent: in neither the networks' training nor the adaptive controller's model)
  after-effects: curl +35 switched off, and the payload dropped, after outward reach 5
Main metric: 3-D distance of the handle from the straight start-target line at peak speed (it also sees a
payload's sag).  Writes <out>/comparison_broad.npz/.json, with episode-0 trajectories for the videos.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from meta import CURL_UNIT, osc_params, per_reach_metrics, rollout, test_episodes
from panda_cortex import CURL
from train import load_params

NONE = jnp.zeros((2, 2))
CONDITIONS = {
    "null": dict(field=NONE),
    "curl +35": dict(field=CURL * CURL_UNIT),
    "curl -35": dict(field=-CURL * CURL_UNIT),
    "mixed": dict(field=jnp.array([[-15.0, 25.0], [-35.0, -8.0]])),
    "push 15 N": dict(field=NONE, const=(0.0, 15.0, 0.0)),
    "payload 1.5 kg": dict(field=NONE, payload=1.5),
    "payload 1.5 kg + curl +35": dict(field=CURL * CURL_UNIT, payload=1.5),
    "curl +70 (2x range)": dict(field=2 * CURL * CURL_UNIT),
    "spring 300 N/m (outside)": dict(field=NONE, const=(0.0, 300.0 * 0.05, 0.0), spring=-300.0 * jnp.eye(2)),
}
AFTER_EFFECTS = {
    "curl +35 switched off": dict(field=CURL * CURL_UNIT),
    "payload 1.5 kg dropped": dict(field=NONE, payload=1.5),
}
VIDEO_CONDITIONS = ("curl +35", "push 15 N", "payload 1.5 kg", "spring 300 N/m (outside)")
N_EPISODES, N_OUT = 32, 6
CRITERION = 0.005          # max 3-D deviation within 5 mm counts as adapted [m]


@partial(jax.jit, static_argnums=0)
def run(kind, params, episode, key):
    traj = rollout(kind, params, episode, key)
    return per_reach_metrics(traj, episode), {k: traj[k][:, 0] for k in ("q", "x", "v")}


def summarize(m):
    out = {k: np.asarray(v)[0::2] for k, v in m.items()}                  # outward reaches, (N_OUT, episodes)
    max_dev = out["max_dev"].mean(1)
    hit = np.flatnonzero(max_dev <= CRITERION)
    return {
        "dev_at_peak_mm": (out["dev_at_peak"].mean(1) * 1e3).round(2).tolist(),
        "max_dev_mm": (max_dev * 1e3).round(2).tolist(),
        "reaches_to_criterion": int(hit[0]) + 1 if len(hit) else None,
        "final_error_mm": float(np.asarray(m["final_error"]).mean() * 1e3),
        "effort": float(np.asarray(m["effort"]).mean()),
        "peak_speed": float(np.asarray(m["peak_speed"])[0::2].mean()),
    }


def tune(name, grid, make, condition, key, noise_key):
    """Pick the gain with the lowest mean deviation on outward reaches 2-6 among stable settings."""
    episode = test_episodes(key, 16, n_out=N_OUT, **condition)
    scores = {}
    for g in grid:
        m, _ = run("osc", make(g), episode, noise_key)
        stable = float(np.asarray(m["final_error"]).mean()) < 0.01
        scores[g] = float(np.asarray(m["dev_at_peak"])[2::2].mean()) if stable else np.inf
        print(f"  {name} {g:8.1f}: {scores[g] * 1e3:6.2f} mm{'' if stable else ' (unstable)'}", flush=True)
    return min(scores, key=scores.get), {f"{g:g}": (s * 1e3 if np.isfinite(s) else None) for g, s in scores.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    k_tune, k_test, k_noise = jax.random.split(jax.random.PRNGKey(2026), 3)

    print("tuning the adaptive controller's gains on separate episodes", flush=True)
    g_v, s_v = tune("velocity gain", (1000.0, 2000.0, 3000.0), lambda g: osc_params(gamma=g),
                    CONDITIONS["curl +35"], k_tune, k_noise)
    g_c, s_c = tune("constant-force gain", (30.0, 100.0, 300.0, 1000.0, 3000.0),
                    lambda g: osc_params(gamma=g_v, gamma_const=g), CONDITIONS["push 15 N"], k_tune, k_noise)
    g_m, s_m = tune("payload-mass gain", (0.3, 1.0, 3.0, 10.0, 30.0),
                    lambda g: osc_params(gamma=g_v, gamma_const=g_c, gamma_mass=g),
                    CONDITIONS["payload 1.5 kg"], k_tune, k_noise)

    controllers = {
        "GRU (broad-trained)": ("gru", load_params(args.out / "meta_gru_broad.npz")),
        "GRU (velocity fields)": ("gru", load_params(args.out / "meta_gru_meta.npz")),
        "MLP (broad-trained)": ("mlp", load_params(args.out / "meta_mlp_broad.npz")),
        "Impedance, compliant": ("osc", osc_params(10.0, 40.0)),
        "Impedance, stiff": ("osc", osc_params(15.0, 120.0)),
        "Adaptive (Slotine-Li)": ("osc", osc_params(10.0, 40.0, g_v, g_c, g_m)),
    }
    results, arrays = {}, {}
    for cond, kwargs in CONDITIONS.items():
        episode = test_episodes(k_test, N_EPISODES, n_out=N_OUT, **kwargs)
        arrays["targets"], arrays["holds"] = np.asarray(episode.target[:, 0]), np.asarray(episode.hold[:, 0])
        results[cond] = {}
        for name, (kind, params) in controllers.items():
            m, traj = run(kind, params, episode, k_noise)
            results[cond][name] = summarize(m)
            for metric, value in m.items():
                arrays[f"{cond}|{name}|{metric}"] = np.asarray(value)
            if cond in VIDEO_CONDITIONS:
                for k, v in traj.items():
                    arrays[f"{cond}|{name}|traj_{k}"] = np.asarray(v)
        print_table(cond, results[cond])

    for cond, kwargs in AFTER_EFFECTS.items():
        episode = test_episodes(k_test, N_EPISODES, n_out=8, off_after=10, **kwargs)
        results[cond] = {}
        print(f"\n== after-effect, {cond} after outward reach 5: 3-D deviation at peak speed per outward reach [mm]")
        for name, (kind, params) in controllers.items():
            m, _ = run(kind, params, episode, k_noise)
            dev = np.asarray(m["dev_at_peak"])[0::2].mean(1) * 1e3
            signed = np.asarray(m["perp_at_peak"])[0::2].mean(1) * 1e3
            results[cond][name] = {"dev_at_peak_mm": dev.round(2).tolist(), "signed_lateral_mm": signed.round(2).tolist()}
            print(f"  {name:28s} " + " ".join(f"{e:5.1f}" for e in dev))

    results["adaptive_gains"] = {"velocity": g_v, "constant": g_c, "mass": g_m,
                                 "scores_mm": {"velocity": s_v, "constant": s_c, "mass": s_m}}
    np.savez_compressed(args.out / "comparison_broad.npz", **arrays)
    (args.out / "comparison_broad.json").write_text(json.dumps(results, indent=1))
    print("\nwrote", args.out / "comparison_broad.npz", "and comparison_broad.json")


def print_table(cond, rows):
    print(f"\n== {cond}: 3-D deviation at peak speed per outward reach [mm] | reaches to <= {CRITERION * 1e3:.0f} mm "
          f"| final error [mm] | effort")
    for name, r in rows.items():
        n = r["reaches_to_criterion"]
        print(f"  {name:28s} " + " ".join(f"{e:5.1f}" for e in r["dev_at_peak_mm"])
              + f"  | {n if n else '>' + str(N_OUT):>3}  | final {r['final_error_mm']:5.1f}  | effort {r['effort']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
