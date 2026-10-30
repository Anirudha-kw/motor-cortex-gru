"""Figure for the longer-training / larger-network comparison.  Reads out/comparison_scale.npz/.json, writes
out/scale_comparison.png: per-reach curves on four conditions and a scorecard of every model on every condition."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import INK, MUTED, OUT

STYLE = {
    "GRU 128 units, 4k steps": ("#F4A582", "-", 1.8),
    "GRU 128 units, 12k steps": ("#D6604D", "-", 2.2),
    "GRU 256 units, 4k steps": ("#B2182B", "-", 2.2),
    "GRU 256 units, 12k steps": ("#67001F", "-", 3.0),
    "GRU velocity fields (specialist)": ("#E69F00", ":", 2.2),
    "Impedance, stiff": ("#0072B2", "--", 1.6),
    "Adaptive (Slotine-Li)": ("#009E73", "--", 1.6),
}
CURVES = (("curl +35", "a   Curl +35"), ("curl +70 (2x range)", "b   Curl +70: 1.4x the trained maximum"),
          ("push 15 N", "c   Constant 15 N push"), ("payload 1.5 kg", "d   Unknown 1.5 kg payload"))
SCORE_COLUMNS = ("curl +35", "curl -35", "mixed", "curl +70 (2x range)", "push 15 N", "payload 1.5 kg",
                 "payload 1.5 kg + curl +35", "spring 300 N/m (outside)")
SHORT = {"curl +70 (2x range)": "curl +70\n(1.4x trained)", "payload 1.5 kg + curl +35": "payload\n+ curl",
         "spring 300 N/m (outside)": "spring\n(outside)", "payload 1.5 kg": "payload", "push 15 N": "push"}


def main():
    results = json.loads((OUT / "comparison_scale.json").read_text())
    names = list(results["curl +35"])

    fig = plt.figure(figsize=(16, 10))
    top = fig.add_gridspec(1, 4, wspace=0.28, left=0.05, right=0.98, top=0.88, bottom=0.6)
    fig.text(0.05, 0.955, "Longer training and a larger network: can the generalist GRU match the specialist?",
             size=17, weight="bold", color=INK)
    fig.text(0.05, 0.922, "Broad-family GRUs (velocity fields + constant forces + payloads) at two sizes and two training "
             "lengths, against the velocity-fields-only specialist and classical control.  Same episodes and noise.",
             size=11, color=MUTED)

    for i, (cond, title) in enumerate(CURVES):
        ax = fig.add_subplot(top[i])
        for name in names:
            color, ls, lw = STYLE[name]
            y = results[cond][name]["headline_mm"]
            ax.plot(np.arange(1, len(y) + 1), y, ls, color=color, lw=lw, marker="o", ms=3.5, label=name)
        metric = results[cond][names[0]]["headline_metric"]
        ax.set_ylabel("|lateral error| at peak speed [mm]" if metric == "perp_at_peak" else "max 3-D deviation [mm]")
        ax.set_xlabel("outward reach")
        ax.set_xticks(range(1, 7))
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", size=11.5, weight="semibold", color=INK)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.488))

    # Scorecard: mean error over outward reaches 2-6 (after the first exposure), and final distance to target.
    ax = fig.add_axes([0.2, 0.06, 0.78, 0.36])
    late = np.array([[np.mean(results[c][n]["headline_mm"][1:]) for c in SCORE_COLUMNS] for n in names])
    final = np.array([[results[c][n]["final_error_mm"] for c in SCORE_COLUMNS] for n in names])
    rank = late / late.min(0, keepdims=True)
    ax.imshow(np.clip(rank, 1, 3), cmap="RdYlGn_r", vmin=1, vmax=3, aspect="auto")
    for i in range(len(names)):
        for j in range(len(SCORE_COLUMNS)):
            ax.text(j, i, f"{late[i, j]:.1f}\n({final[i, j]:.1f})", ha="center", va="center", size=8.5,
                    color=INK, weight="semibold" if rank[i, j] < 1.05 else "normal")
    ax.set_xticks(range(len(SCORE_COLUMNS)))
    ax.set_xticklabels([SHORT.get(c, c) for c in SCORE_COLUMNS], size=9.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, size=10)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("e   Mean error on outward reaches 2-6 [mm] (final distance to target in brackets);  "
                 "green = best in column, red = 3x the best or worse", loc="left", size=11, weight="semibold",
                 color=INK, x=-0.17)

    fig.savefig(OUT / "scale_comparison.png", dpi=150)
    print("wrote", OUT / "scale_comparison.png")


if __name__ == "__main__":
    main()
