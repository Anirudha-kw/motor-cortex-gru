"""Figure for the broad-family comparison.  Reads out/comparison_broad.npz/.json, writes
out/broad_comparison.png.  Velocity fields are scored by horizontal lateral error; forces, payloads and the
spring by the maximum 3-D distance from the straight path (which also sees a payload's sag)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figures import INK, MUTED, OUT

STYLE = {
    "GRU (broad-trained)": ("#D55E00", "-", 3.0),
    "GRU (velocity fields)": ("#E69F00", "-", 1.8),
    "MLP (broad-trained)": ("#CC79A7", "-", 1.8),
    "Impedance, compliant": ("#56B4E9", "--", 1.6),
    "Impedance, stiff": ("#0072B2", "--", 1.8),
    "Adaptive (Slotine-Li)": ("#009E73", "--", 1.8),
}
SHOWN = ("GRU (broad-trained)", "GRU (velocity fields)", "MLP (broad-trained)", "Impedance, stiff", "Adaptive (Slotine-Li)")


def per_reach(arrays, cond, name, metric):
    values = arrays[f"{cond}|{name}|{metric}"][0::2]                   # outward reaches, (6, episodes)
    return np.abs(values).mean(1) * 1e3


def curves(ax, arrays, cond, metric, title, ylabel=None):
    for name in SHOWN:
        color, ls, lw = STYLE[name]
        y = per_reach(arrays, cond, name, metric)
        ax.plot(np.arange(1, len(y) + 1), y, ls, color=color, lw=lw, marker="o", ms=4)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("outward reach")
    ax.set_ylim(bottom=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", size=11.5, weight="semibold", color=INK)


def bars(ax, results, conds, key, title, ylabel, log=False):
    width = 0.8 / len(SHOWN)
    for i, name in enumerate(SHOWN):
        color = STYLE[name][0]
        ax.bar(np.arange(len(conds)) + (i - (len(SHOWN) - 1) / 2) * width, [results[c][name][key] for c in conds],
               width, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(conds)))
    short = {"push 15 N": "push", "payload 1.5 kg": "payload", "payload 1.5 kg + curl +35": "payload\n+ curl",
             "spring 300 N/m (outside)": "spring\n(outside)", "curl +35": "curl +35"}
    ax.set_xticklabels([short[c] for c in conds], size=9.5)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", size=11.5, weight="semibold", color=INK)


def main():
    results = json.loads((OUT / "comparison_broad.json").read_text())
    arrays = dict(np.load(OUT / "comparison_broad.npz"))

    fig = plt.figure(figsize=(16, 9.4))
    outer = fig.add_gridspec(2, 1, hspace=0.45, left=0.05, right=0.98, top=0.87, bottom=0.13)
    top = outer[0].subgridspec(1, 4, wspace=0.3)
    bottom = outer[1].subgridspec(1, 4, wspace=0.32)
    fig.text(0.05, 0.955, "Training on constant forces and payloads fixes the failure case, at a cost on strong "
             "velocity fields", size=17, weight="bold", color=INK)
    fig.text(0.05, 0.922, "Frozen controllers on the Panda, identical episodes, delays, noise and torque limits.  "
             "The adaptive controller is given the exact structure of velocity fields, constant forces and payloads.",
             size=11, color=MUTED)

    curves(fig.add_subplot(top[0]), arrays, "push 15 N", "max_dev", "a   Constant 15 N push (old failure)",
           "max 3-D deviation from straight path [mm]")
    curves(fig.add_subplot(top[1]), arrays, "payload 1.5 kg", "max_dev", "b   Unknown 1.5 kg payload")
    curves(fig.add_subplot(top[2]), arrays, "curl +35", "perp_at_peak", "c   Curl +35 (velocity field)",
           "|lateral error at peak speed| [mm]")
    curves(fig.add_subplot(top[3]), arrays, "spring 300 N/m (outside)", "max_dev",
           "d   Spring field: outside everything")

    bars(fig.add_subplot(bottom[0]), results, ("push 15 N", "payload 1.5 kg", "payload 1.5 kg + curl +35",
                                                "spring 300 N/m (outside)"),
         "final_error_mm", "e   Where the hand ends up", "final distance to target [mm]", log=True)
    bars(fig.add_subplot(bottom[1]), results, ("curl +35", "push 15 N", "payload 1.5 kg"), "effort",
         "f   Effort", "mean squared torque / limit")
    curves(fig.add_subplot(bottom[2]), arrays, "curl +70 (2x range)", "perp_at_peak",
           "g   Curl +70: 1.4x the trained maximum", "|lateral error at peak speed| [mm]")

    ax = fig.add_subplot(bottom[3])
    for name in SHOWN:
        color, ls, lw = STYLE[name]
        y = results["payload 1.5 kg dropped"][name]["dev_at_peak_mm"]
        ax.plot(np.arange(1, len(y) + 1), y, ls, color=color, lw=lw, marker="o", ms=4)
    ax.axvline(5.5, color=MUTED, lw=1, ls=":")
    ax.text(5.6, ax.get_ylim()[1] * 0.92, "payload dropped", size=9.5, color=MUTED)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("outward reach")
    ax.set_ylabel("3-D deviation at peak speed [mm]")
    ax.set_ylim(bottom=0)
    ax.set_title("h   Payload dropped after reach 5", loc="left", size=11.5, weight="semibold", color=INK)

    handles = [plt.Line2D([], [], color=STYLE[n][0], ls=STYLE[n][1], lw=2.4, marker="o", ms=4) for n in SHOWN]
    gains = results["adaptive_gains"]
    labels = [n if not n.startswith("Adaptive") else
              f"{n} (gains {gains['velocity']:g} / {gains['constant']:g} / {gains['mass']:g})" for n in SHOWN]
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "broad_comparison.png", dpi=150)
    print("wrote", OUT / "broad_comparison.png")


if __name__ == "__main__":
    main()
