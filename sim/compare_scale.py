"""Does a longer training run or a larger network recover the specialist's velocity-field adaptation
without losing the broad family's robustness?

Broad-family GRUs (velocity fields + constant forces + payloads):
  128 units x 4000 steps (meta_gru_broad.npz), 128 x 12000, 256 x 4000, 256 x 12000
References: the velocity-fields-only GRU (the specialist), stiff impedance, and the adaptive controller with the
gains tuned in compare_broad.py.  Same conditions, episodes and noise as compare_broad.py.

Writes <out>/comparison_scale.npz and <out>/comparison_scale.json and prints the tables.
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import compare_broad as cb
from meta import osc_params, test_episodes
from train import load_params

MODELS = {
    "GRU 128 units, 4k steps": "meta_gru_broad.npz",
    "GRU 128 units, 12k steps": "meta_gru_broad_h128_12k.npz",
    "GRU 256 units, 4k steps": "meta_gru_broad_h256_4k.npz",
    "GRU 256 units, 12k steps": "meta_gru_broad_h256_12k.npz",
    "GRU velocity fields (specialist)": "meta_gru_meta.npz",
}
VELOCITY_FIELDS = ("null", "curl +35", "curl -35", "mixed", "curl +70 (2x range)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out")
    parser.add_argument("--gains", type=float, nargs=3, default=(2000.0, 300.0, 0.3),
                        help="adaptive gains (velocity, constant force, payload mass) from compare_broad.py")
    args = parser.parse_args()
    _, k_test, k_noise = jax.random.split(jax.random.PRNGKey(2026), 3)       # as in compare_broad.py

    controllers = {name: ("gru", load_params(args.out / f)) for name, f in MODELS.items()}
    controllers["Impedance, stiff"] = ("osc", osc_params(15.0, 120.0))
    controllers["Adaptive (Slotine-Li)"] = ("osc", osc_params(10.0, 40.0, *args.gains))

    results, arrays = {}, {}
    for cond, kwargs in cb.CONDITIONS.items():
        episode = test_episodes(k_test, cb.N_EPISODES, n_out=cb.N_OUT, **kwargs)
        results[cond] = {}
        metric = "perp_at_peak" if cond in VELOCITY_FIELDS else "max_dev"
        label = "horizontal |lateral error| at peak speed" if metric == "perp_at_peak" else "max 3-D deviation"
        print(f"\n== {cond}: {label}, outward reach 1 -> 2 -> 6 [mm] | final error [mm] | effort x1e3", flush=True)
        for name, (kind, params) in controllers.items():
            m, _ = cb.run(kind, params, episode, k_noise)
            results[cond][name] = cb.summarize(m)
            curve = np.abs(np.asarray(m[metric]))[0::2].mean(1) * 1e3
            results[cond][name]["headline_mm"] = curve.round(2).tolist()
            results[cond][name]["headline_metric"] = metric
            for k, v in m.items():
                arrays[f"{cond}|{name}|{k}"] = np.asarray(v)
            print(f"  {name:34s} {curve[0]:5.1f} -> {curve[1]:5.1f} -> {curve[-1]:5.1f}"
                  f"  | final {results[cond][name]['final_error_mm']:5.1f}  | effort {results[cond][name]['effort'] * 1e3:5.1f}",
                  flush=True)

    for cond, kwargs in cb.AFTER_EFFECTS.items():
        episode = test_episodes(k_test, cb.N_EPISODES, n_out=8, off_after=10, **kwargs)
        results[cond] = {}
        print(f"\n== after-effect, {cond} after outward reach 5: 3-D deviation at peak speed per outward reach [mm]")
        for name, (kind, params) in controllers.items():
            m, _ = cb.run(kind, params, episode, k_noise)
            dev = np.asarray(m["dev_at_peak"])[0::2].mean(1) * 1e3
            signed = np.asarray(m["perp_at_peak"])[0::2].mean(1) * 1e3
            results[cond][name] = {"dev_at_peak_mm": dev.round(2).tolist(), "signed_lateral_mm": signed.round(2).tolist()}
            print(f"  {name:34s} " + " ".join(f"{e:5.1f}" for e in dev), flush=True)

    results["adaptive_gains"] = dict(zip(("velocity", "constant", "mass"), args.gains))
    np.savez_compressed(args.out / "comparison_scale.npz", **arrays)
    (args.out / "comparison_scale.json").write_text(json.dumps(results, indent=1))
    print("\nwrote", args.out / "comparison_scale.npz", "and comparison_scale.json")


if __name__ == "__main__":
    main()
