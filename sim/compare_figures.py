"""Figure for the in-context adaptation comparison.  Reads out/comparison.npz and out/comparison.json,
writes out/incontext_comparison.png."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle

from figures import INK, MUTED, OUT

REACH_STEPS = 70          # meta.REACH_STEPS (not imported: that would build the MJX model)
STYLE = {                 # name prefix -> colour, line style
    "GRU (meta-trained)": ("#D55E00", "-"),
    "GRU (no fields)": ("#8C959F", "-"),
    "MLP (meta-trained)": ("#CC79A7", "-"),
    "Impedance, compliant": ("#56B4E9", "--"),
    "Impedance, stiff": ("#0072B2", "--"),
    "Adaptive": ("#009E73", "--"),
}
PATH_PANELS = ("GRU (meta-trained)", "MLP (meta-trained)", "Impedance, stiff", "Adaptive")


def style(name):
    return next(v for k, v in STYLE.items() if name.startswith(k))


def find(names, prefix):
    return next(n for n in names if n.startswith(prefix))


def outward_paths(arrays, field, name):
    path, targets, holds = arrays[f"{field}|{name}|path"], arrays["targets"], arrays["holds"]
    home = holds[0]
    top = lambda p: np.stack([-(p[..., 1] - home[1]), p[..., 0] - home[0]], -1) * 100      # cm, forward up
    reaches = range(0, len(path) // REACH_STEPS, 2)
    return ([top(path[r * REACH_STEPS:(r + 1) * REACH_STEPS]) for r in reaches],
            [top(targets[r * REACH_STEPS]) for r in reaches])


def curve(ax, results, field, metric="lateral_at_peak_mm"):
    for name, r in results[field].items():
        color, ls = style(name)
        lw = 2.8 if name.startswith("GRU (meta") else 1.8
        ax.plot(np.arange(1, len(r[metric]) + 1), r[metric], ls, color=color, lw=lw, marker="o", ms=4, label=name)
    ax.set_xlabel("outward reach in the new field")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(bottom=0)


def main():
    results = json.loads((OUT / "comparison.json").read_text())
    arrays = dict(np.load(OUT / "comparison.npz"))
    names = list(results["curl +35"])

    fig = plt.figure(figsize=(16, 9.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.42, left=0.05, right=0.98, top=0.87, bottom=0.12)
    top = outer[0].subgridspec(1, 4, wspace=0.08)
    bottom = outer[1].subgridspec(1, 4, wspace=0.32)
    fig.text(0.05, 0.955, "In-context adaptation on a 7-DoF Panda: a meta-trained GRU vs classical control",
             size=17, weight="bold", color=INK)
    fig.text(0.05, 0.922, "All controllers frozen; only the GRU's hidden state and the adaptive controller's estimates "
             "change.  Same episodes, same 10 ms sensing delay, motor lag, noise and torque limits.", size=11,
             color=MUTED)

    for i, prefix in enumerate(PATH_PANELS):
        name = find(names, prefix)
        ax = fig.add_subplot(top[i])
        segments, targets = outward_paths(arrays, "curl +35", name)
        base = np.array(to_rgb(style(name)[0]))
        for k, (seg, target) in enumerate(zip(segments, targets)):
            shade = 0.25 + 0.75 * k / (len(segments) - 1)
            ax.plot(seg[:, 0], seg[:, 1], color=1 - shade * (1 - base), lw=2.2)
            ax.add_patch(Circle(target, 0.9, fc="none", ec="#8C959F", lw=1.2))
            ax.text(*(target * 1.22), str(k + 1), ha="center", va="center", size=9, color=MUTED)
        ax.add_patch(Circle((0, 0), 0.4, fc=INK, ec="none"))
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        r = results["curl +35"][name]
        ax.set_title(f"{'abcd'[i]}   {name}", loc="left", size=11.5, weight="semibold", color=INK)
        ax.set_xlabel(f"reach 1: {r['lateral_at_peak_mm'][0]:.1f} mm  →  reach 6: {r['lateral_at_peak_mm'][-1]:.1f} mm",
                      size=9.5, color=MUTED)
    fig.text(0.05, 0.505, "Curl field +35 N·s/m, handle paths of outward reaches 1 (light) to 6 (dark), seen from above",
             size=10, color=MUTED)

    ax = fig.add_subplot(bottom[0])
    curve(ax, results, "curl +35")
    ax.set_ylabel("|lateral error at peak speed| [mm]")
    ax.set_title("e   Inside the training family", loc="left", size=11.5, weight="semibold", color=INK)
    ax.text(0.98, 0.95, "curl +35", transform=ax.transAxes, ha="right", va="top", size=9.5, color=MUTED)

    ax = fig.add_subplot(bottom[1])
    curve(ax, results, "constant 15 N")
    ax.set_title("f   Outside it: constant 15 N push", loc="left", size=11.5, weight="semibold", color=INK)

    ax = fig.add_subplot(bottom[2])
    for name, r in results["after-effect"].items():
        color, ls = style(name)
        signed = r["signed_lateral_at_peak_mm"]
        ax.plot(np.arange(1, len(signed) + 1), signed, ls, color=color, lw=2.8 if name.startswith("GRU (meta") else 1.8,
                marker="o", ms=4)
    ax.axvline(5.5, color=MUTED, lw=1, ls=":")
    ax.axhline(0, color="#C9D1D9", lw=1)
    ax.text(5.6, ax.get_ylim()[1] * 0.92, "field off", size=9.5, color=MUTED)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("outward reach")
    ax.set_ylabel("signed lateral error at peak speed [mm]")
    ax.set_title("g   After-effect when the field stops", loc="left", size=11.5, weight="semibold", color=INK)

    ax = fig.add_subplot(bottom[3])
    for name, r in results["curl +35"].items():
        color, _ = style(name)
        late = np.mean(r["max_lateral_mm"][1:])
        ax.scatter(r["effort"], late, s=90, color=color, edgecolor="white", lw=1.5, zorder=3)
    ax.set_xlabel("effort (mean squared torque / limit)")
    ax.set_ylabel("max lateral error, reaches 2-6 [mm]")
    ax.set_ylim(bottom=0)
    ax.set_title("h   Accuracy vs effort (curl +35)", loc="left", size=11.5, weight="semibold", color=INK)
    handles = [plt.Line2D([], [], color=style(n)[0], ls=style(n)[1], lw=2.4, marker="o", ms=4) for n in names]
    fig.legend(handles, names, loc="lower center", ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(OUT / "incontext_comparison.png", dpi=150)
    print("wrote", OUT / "incontext_comparison.png")


if __name__ == "__main__":
    main()
