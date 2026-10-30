"""Summary figure: hand paths, speed profiles, learning curves, and channel-trial force profiles.

Reads out/adaptation.npz (human arm) or, with --robot panda, out/panda_adaptation.npz, writes the
matching PNG and prints the numbers it plots, plus a jPCA test for rotational population dynamics
(Churchland et al. 2012) that is reported, not plotted.
"""

import argparse

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from arm import CURL, DT

OUT = Path(__file__).resolve().parent / "out"
CONDITIONS = (("baseline", "Null field: baseline"), ("early_field", "Curl field: first exposure"),
              ("late_field", "Curl field: after learning"), ("after_effect", "Field removed: after-effect"))
INK, MUTED, FIELD_BG, WASH_BG = "#1F2328", "#656D76", "#FDF1E7", "#EAF2FB"
C_FIELD, C_NULL, C_AI = "#D55E00", "#0072B2", "#009E73"

for font in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
    font_manager.fontManager.addfont(str(font))
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "mathtext.fontset": "stixsans", "font.size": 10.5,
    "axes.edgecolor": "#C9D1D9", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def target_colors(n):
    return plt.cm.twilight_shifted(np.linspace(0, 1, n, endpoint=False))


def onset_index(speed):
    """Movement onset per reach: first step above 5% of peak speed.  speed: (T, B)."""
    return np.argmax(speed > 0.05 * speed.max(0), axis=0)


def jpca(h, onsets, before=5, after=20, n_pcs=6):
    """Churchland et al. (2012): fit dX/dt = X M with M skew-symmetric in the top PCs around movement onset.

    h: (T, C, N) hidden states.  Returns projections onto the leading jPC plane (C, W, 2) and R^2 values.
    """
    X = np.stack([h[o - before:o + after, c] for c, o in enumerate(onsets)])         # (C, W, N)
    X = X - X.mean(0, keepdims=True)                                                   # remove the condition mean
    flat = X.reshape(-1, X.shape[-1])
    _, _, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
    Z = (X - flat.mean(0)) @ Vt[:n_pcs].T                                              # (C, W, k)
    dZ = (Z[:, 2:] - Z[:, :-2]) / (2 * DT)
    Zm = Z[:, 1:-1].reshape(-1, n_pcs)
    dZ = dZ.reshape(-1, n_pcs)

    M_full = np.linalg.lstsq(Zm, dZ, rcond=None)[0]
    iu = np.triu_indices(n_pcs, 1)                                                     # skew-symmetric basis
    basis = []
    for i, j in zip(*iu):
        B = np.zeros((n_pcs, n_pcs))
        B[i, j], B[j, i] = 1.0, -1.0
        basis.append((Zm @ B).ravel())
    k = np.linalg.lstsq(np.stack(basis, 1), dZ.ravel(), rcond=None)[0]
    M_skew = np.zeros((n_pcs, n_pcs))
    M_skew[iu] = k
    M_skew -= M_skew.T

    r2 = lambda M: 1 - np.sum((dZ - Zm @ M) ** 2) / np.sum((dZ - dZ.mean(0)) ** 2)
    evals, evecs = np.linalg.eig(M_skew)
    v = evecs[:, np.argmax(np.abs(evals.imag))]
    plane = np.linalg.qr(np.stack([v.real, v.imag], 1))[0]
    return Z @ plane, r2(M_full), r2(M_skew), np.abs(evals.imag).max() / (2 * np.pi)


PLANTS = {
    "arm": ("adaptation.npz", "forcefield_adaptation.png",
            "A GRU 'motor cortex' adapts to a force field the way people do",
            "Two-joint, six-muscle arm with 30 ms proprioceptive and 70 ms visual delays; the network is trained "
            "by backpropagating task error and muscle effort through the arm's physics."),
    "panda": ("panda_adaptation.npz", "panda_forcefield_adaptation.png",
              "The same GRU 'motor cortex' on a 7-DoF Franka Panda: force-field adaptation",
              "Torque-controlled Panda in MuJoCo MJX (gravity compensated, 10 ms sensing delay); the network is "
              "trained by backpropagating task error and torque effort through the robot's physics."),
}


def load(plant):
    """Results as a dict, reduced to the horizontal plane (the Panda's data are 3-D)."""
    name, png, title, subtitle = PLANTS[plant]
    d = {}
    for key, value in np.load(OUT / name).items():
        if key.endswith(("_x", "_v", "_channel_force")) or key in ("start", "target"):
            value = value[..., :2]
        d[key] = value
    d.setdefault("curl", CURL)
    return d, OUT / png, title, subtitle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=sorted(PLANTS), default="arm")
    d, png, title, subtitle = load(parser.parse_args().robot)
    start, target = d["start"], d["target"]
    n = len(target)
    colors = target_colors(n)
    t_ms = (np.arange(d["baseline_x"].shape[0]) + 1) * DT * 1e3

    fig = plt.figure(figsize=(15, 8.6))
    outer = fig.add_gridspec(2, 1, hspace=0.38, left=0.05, right=0.97, top=0.88, bottom=0.08)
    top = outer[0].subgridspec(1, 4, wspace=0.12)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1, 2.1, 1], wspace=0.45)
    fig.text(0.05, 0.955, title, size=17, weight="bold", color=INK)
    fig.text(0.05, 0.922, subtitle, size=11, color=MUTED)

    # Row 1: hand paths in the four conditions
    centre = start[0]
    for i, (key, title) in enumerate(CONDITIONS):
        ax = fig.add_subplot(top[i])
        x = d[f"{key}_x"]
        for c in range(n):
            ax.plot(*(np.vstack([start[c], x[:, c]]) - centre).T * 100, color=colors[c], lw=1.8)
            ax.add_patch(plt.Circle((target[c] - centre) * 100, 0.6, fc="none", ec=MUTED, lw=1))
        ax.add_patch(plt.Circle((0, 0), 0.5, fc=INK, ec="none"))
        ax.set_title(f"{'abcd'[i]}   {title}", loc="left", size=11.5, weight="semibold", color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        perp = np.abs(_max_perp(x, start, target)).mean() * 1e3
        end = np.linalg.norm(x[-1] - target, axis=-1).mean() * 1e3
        ax.set_xlabel(f"max lateral error {perp:.0f} mm  ·  final error {end:.1f} mm", size=9.5, color=MUTED)
        if i == 0:
            ax.plot([-13, -8], [-13, -13], color=INK, lw=2)
            ax.text(-10.5, -12.3, "5 cm", ha="center", size=9, color=INK)
        if i in (1, 2):
            ax.text(13.5, -13.5, "curl field on", ha="right", size=9.5, color=C_FIELD, weight="semibold")

    # e: speed profiles at baseline
    ax = fig.add_subplot(bottom[0])
    speed = np.linalg.norm(d["baseline_v"], axis=-1)
    for c in range(n):
        ax.plot(t_ms - d["go"][c] * DT * 1e3, speed[:, c], color=colors[c], lw=1.6)
    ax.set_xlim(-50, 600)
    ax.set_xlabel("time from go cue [ms]")
    ax.set_ylabel("hand speed [m/s]")
    ax.set_title("e   Bell-shaped speed profiles (baseline)", loc="left", size=11.5, weight="semibold", color=INK)

    # f: learning and washout curves
    ax = fig.add_subplot(bottom[1])
    nf, nw = int(d["updates_field"]), int(d["updates_washout"])
    upd = np.arange(nf + nw + 1)
    error = np.concatenate([d["curve_perp_field"][:nf + 1], d["curve_perp_null"][nf + 1:]]) * 1e3
    ax.axvspan(0.5, nf + 0.5, color=FIELD_BG, lw=0)
    ax.axvspan(nf + 0.5, nf + nw, color=WASH_BG, lw=0)
    ax.axhline(0, color="#C9D1D9", lw=1)
    ax.plot(upd[: nf + 1], error[: nf + 1], color=C_FIELD, lw=2, label="lateral error, curl field")
    ax.plot(upd[nf:], error[nf:], color=C_NULL, lw=2, label="lateral error, null field")
    ax.set_xlim(0, nf + nw)
    ax.set_xlabel("learning updates (gradient steps on the network's weights)")
    ax.set_ylabel("lateral error at peak speed [mm]")
    ax.text(nf / 2, ax.get_ylim()[1] * 0.92, "curl field on", ha="center", size=10, color=C_FIELD, weight="semibold")
    ax.text(nf + nw / 2, ax.get_ylim()[1] * 0.92, "washout (field off)", ha="center", size=10, color=C_NULL,
            weight="semibold")
    ax2 = ax.twinx()
    ax2.plot(upd, d["curve_adaptation_index"], color=C_AI, lw=2, ls=(0, (4, 2)), label="adaptation index (channel)")
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_ylabel("adaptation index", color=C_AI)
    ax2.spines["right"].set_visible(True)
    ax2.tick_params(axis="y", colors=C_AI)
    lines = ax.get_lines()[1:] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("f   Adaptation, after-effect and washout", loc="left", size=11.5, weight="semibold", color=INK)

    # g: force-channel probes -- a predictive internal model, not just a stiffer arm
    ax = fig.add_subplot(bottom[2])
    t_go = t_ms - d["go"][0] * DT * 1e3
    profiles = {}
    for key, label, color in (("baseline_channel", "before learning", "#8C959F"),
                              ("late_channel", "after learning", C_FIELD)):
        produced, ideal = channel_profiles(d, key)
        profiles[key] = (produced, ideal)
        ax.plot(t_go, produced, color=color, lw=2, label=label)
    ax.plot(t_go, profiles["late_channel"][1], color=INK, lw=1.5, ls=(0, (4, 2)),
            label="ideal")
    ax.axhline(0, color="#C9D1D9", lw=1)
    ax.set_xlim(-50, 600)
    ax.set_xlabel("time from go cue [ms]")
    ax.set_ylabel("force into channel wall [N]")
    ax.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.6)
    ax.set_title("g   Channel trials: learned force profile", loc="left", size=11.5, weight="semibold", color=INK)

    fig.savefig(png, dpi=150)
    m = speed.max(0)
    print(f"baseline: peak speed {m.mean():.2f} m/s (range {m.min():.2f}-{m.max():.2f})")
    for key, title in CONDITIONS:
        print(f"{title:32s} mean |max lateral error| = {np.abs(_max_perp(d[f'{key}_x'], start, target)).mean() * 1e3:5.1f} mm,"
              f" endpoint error = {np.linalg.norm(d[f'{key}_x'][-1] - target, axis=-1).mean() * 1e3:4.1f} mm")
    print(f"adaptation index: start {d['curve_adaptation_index'][0]:.2f}, end of field {d['curve_adaptation_index'][nf]:.2f},"
          f" end of washout {d['curve_adaptation_index'][-1]:.2f}")
    print(f"lateral error at peak speed: first field exposure {d['curve_perp_field'][0] * 1e3:+.1f} mm, end of field "
          f"{d['curve_perp_field'][nf] * 1e3:+.1f} mm, after-effect {d['curve_perp_null'][nf] * 1e3:+.1f} mm, end of washout "
          f"{d['curve_perp_null'][-1] * 1e3:+.1f} mm")
    if "curve_cocontraction_field" in d:      # muscles only; the Panda has no antagonist pairs
        print(f"co-contraction (field probes): start {d['curve_cocontraction_field'][0]:.4f}, max "
              f"{d['curve_cocontraction_field'][:nf + 1].max():.4f}, end of field {d['curve_cocontraction_field'][nf]:.4f}")
    produced, ideal = profiles["late_channel"]
    print(f"channel force after learning: peak {produced.max():.2f} N vs ideal {ideal.max():.2f} N, "
          f"correlation {np.corrcoef(produced, ideal)[0, 1]:.3f}")
    proj, r2_full, r2_skew, freq = jpca(d["baseline_h"], onset_index(speed))
    print(f"jPCA (baseline network): skew-symmetric R^2 = {r2_skew:.2f}, unconstrained R^2 = {r2_full:.2f}, "
          f"ratio {r2_skew / r2_full:.2f}, rotation frequency {freq:.1f} Hz")
    print("wrote", png)


def channel_profiles(d, key):
    """Mean lateral force the arm pushes into the channel, and the force that would cancel the curl field."""
    start, target = d["start"], d["target"]
    direction = (target - start) / np.linalg.norm(target - start, axis=-1, keepdims=True)
    normal = np.stack([-direction[:, 1], direction[:, 0]], -1)
    v = d[f"{key}_v"]
    field_force = float(d["curl"]) * np.stack([v[..., 1], -v[..., 0]], -1)
    ideal = -np.sum(field_force * normal[None], -1).mean(1)
    produced = -np.sum(d[f"{key}_channel_force"] * normal[None], -1).mean(1)
    return produced, ideal


def _max_perp(x, start, target):
    direction = (target - start) / np.linalg.norm(target - start, axis=-1, keepdims=True)
    normal = np.stack([-direction[:, 1], direction[:, 0]], -1)
    perp = np.sum((x - start[None]) * normal[None], -1)
    return perp[np.abs(perp).argmax(0), np.arange(perp.shape[1])]


if __name__ == "__main__":
    main()
