"""Head-to-head on the Panda: in-context adaptation vs memoryless, untrained-for-fields and classical control.

Every controller is frozen during the test; only the GRU's hidden state and the adaptive controller's
parameter estimates evolve.  All face identical episodes (same target orders, same sensor/motor noise):
  GRU (meta-trained)      meta_gru_meta.npz: random force fields, hidden state carried across reaches
  GRU (no fields)         meta_gru_null.npz: same architecture and episodes, never saw a field
  MLP (meta-trained)      meta_mlp_meta.npz: same training as the meta GRU, but no memory
  Impedance, compliant    task-space impedance (400 N/m, 40 N s/m) tracking a minimum-jerk path
  Impedance, stiff        the same at 1800 N/m, 120 N s/m
  Adaptive (Slotine-Li)   compliant impedance + adaptive velocity-field compensation (the field's exact
                          structure is built in; adaptation gain tuned on separate curl +35 episodes)
Fields: null; curl +35 and -35; a mixed curl + viscous field (all inside the training family); curl +70
(twice the test field, 1.4x the trained maximum of 50); a constant 15 N sideways push (outside the training
family and outside the adaptive controller's model).  After-effect episodes: curl +35 for 5 outward reaches, then
off.

Writes <out>/comparison.npz and <out>/comparison.json and prints the tables.
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

FIELDS = {
    "null": (jnp.zeros((2, 2)), (0.0, 0.0, 0.0)),
    "curl +35": (CURL * CURL_UNIT, (0.0, 0.0, 0.0)),
    "curl -35": (-CURL * CURL_UNIT, (0.0, 0.0, 0.0)),
    "mixed": (jnp.array([[-15.0, 25.0], [-35.0, -8.0]]), (0.0, 0.0, 0.0)),     # curl 30 + viscous (-5.4, -17.6)
    "curl +70 (2x range)": (2 * CURL * CURL_UNIT, (0.0, 0.0, 0.0)),   # key used by the result files
    "constant 15 N": (jnp.zeros((2, 2)), (0.0, 15.0, 0.0)),
}
GAMMAS = (300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 1e4)
N_EPISODES = 32
N_OUT = 6
CRITERION = 0.005          # |max lateral error| within 5 mm counts as adapted [m]


@partial(jax.jit, static_argnums=0)
def evaluate(kind, params, episode, key):
    return per_reach_metrics(rollout(kind, params, episode, key), episode)


@partial(jax.jit, static_argnums=0)
def paths(kind, params, episode, key):
    return rollout(kind, params, episode, key)["x"][:, 0]                 # first episode, (T, 3)


def summarize(m):
    """Episode-averaged summary of one controller in one field (outward reaches only)."""
    out = {k: np.asarray(v)[0::2] for k, v in m.items()}                  # (N_OUT, episodes)
    lateral = np.abs(out["perp_at_peak"]).mean(1)
    max_perp = out["max_perp"].mean(1)
    hit = np.flatnonzero(max_perp <= CRITERION)
    return {
        "lateral_at_peak_mm": (lateral * 1e3).round(2).tolist(),
        "max_lateral_mm": (max_perp * 1e3).round(2).tolist(),
        "reaches_to_criterion": int(hit[0]) + 1 if len(hit) else None,
        "final_error_mm": float(np.asarray(m["final_error"]).mean() * 1e3),
        "effort": float(np.asarray(m["effort"]).mean()),
        "peak_speed": float(np.asarray(m["peak_speed"])[0::2].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    args = parser.parse_args()
    key = jax.random.PRNGKey(2026)
    k_tune, k_test, k_noise = jax.random.split(key, 3)

    # Tune the adaptive controller's gain on separate curl +35 episodes.
    tune_episode = test_episodes(k_tune, 16, *FIELDS["curl +35"], n_out=N_OUT)
    scores = {}
    for gamma in GAMMAS:
        m = evaluate("osc", osc_params(gamma=gamma), tune_episode, k_noise)
        stable = float(np.asarray(m["final_error"]).mean()) < 0.01
        scores[gamma] = float(np.abs(np.asarray(m["perp_at_peak"])[2::2]).mean()) if stable else np.inf
        print(f"adaptive gain {gamma:8.0f}: mean |lateral error| over outward reaches 2-{N_OUT} "
              f"{scores[gamma] * 1e3:.2f} mm{'' if stable else ' (unstable)'}", flush=True)
    gamma = min(scores, key=scores.get)

    controllers = {
        "GRU (meta-trained)": ("gru", load_params(args.out / "meta_gru_meta.npz")),
        "GRU (no fields)": ("gru", load_params(args.out / "meta_gru_null.npz")),
        "MLP (meta-trained)": ("mlp", load_params(args.out / "meta_mlp_meta.npz")),
        "Impedance, compliant": ("osc", osc_params(10.0, 40.0)),
        "Impedance, stiff": ("osc", osc_params(15.0, 120.0)),
        f"Adaptive (Slotine-Li, gain {gamma:.0e})": ("osc", osc_params(10.0, 40.0, gamma)),
    }
    results, arrays = {}, {}
    for field_name, (field, const) in FIELDS.items():
        episode = test_episodes(k_test, N_EPISODES, field, const, n_out=N_OUT)
        arrays["targets"], arrays["holds"] = np.asarray(episode.target[:, 0]), np.asarray(episode.hold[:, 0])
        results[field_name] = {}
        for name, (kind, params) in controllers.items():
            m = evaluate(kind, params, episode, k_noise)
            results[field_name][name] = summarize(m)
            for metric, value in m.items():
                arrays[f"{field_name}|{name}|{metric}"] = np.asarray(value)
            if field_name in ("curl +35", "constant 15 N"):
                arrays[f"{field_name}|{name}|path"] = np.asarray(paths(kind, params, episode, k_noise))
        print_table(field_name, results[field_name])

    after = test_episodes(k_test, N_EPISODES, *FIELDS["curl +35"], n_out=8, off_after=10)
    results["after-effect"] = {}
    print("\nafter-effect: curl +35 for outward reaches 1-5, off for 6-8 -- signed lateral error at peak speed [mm]")
    for name, (kind, params) in controllers.items():
        m = evaluate(kind, params, after, k_noise)
        signed = np.asarray(m["perp_at_peak"])[0::2].mean(1) * 1e3
        results["after-effect"][name] = {"signed_lateral_at_peak_mm": signed.round(2).tolist()}
        arrays[f"after-effect|{name}|perp_at_peak"] = np.asarray(m["perp_at_peak"])
        print(f"  {name:38s} " + " ".join(f"{e:+6.1f}" for e in signed))

    results["adaptive_gain_scores_mm"] = {f"{g:.0e}": (s * 1e3 if np.isfinite(s) else None) for g, s in scores.items()}
    np.savez_compressed(args.out / "comparison.npz", **arrays)
    (args.out / "comparison.json").write_text(json.dumps(results, indent=1))
    print("\nwrote", args.out / "comparison.npz", "and comparison.json")


def print_table(field_name, rows):
    print(f"\n== {field_name}: |lateral error at peak speed| per outward reach [mm], reaches to <= "
          f"{CRITERION * 1e3:.0f} mm max lateral error, final error [mm], effort")
    for name, r in rows.items():
        n = r["reaches_to_criterion"]
        print(f"  {name:38s} " + " ".join(f"{e:5.1f}" for e in r["lateral_at_peak_mm"])
              + f"  | criterion: {n if n else '>' + str(N_OUT):>3}  | final {r['final_error_mm']:5.1f}"
              f"  | effort {r['effort']:.4f}  | peak speed {r['peak_speed']:.2f} m/s")


if __name__ == "__main__":
    main()
