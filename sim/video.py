"""Video of the arm reaching in the four stages of the experiment, side by side, in real time.

Top view of the two-joint arm and its hand paths; in the curl-field stages an orange arrow shows the
field's push on the hand.  Reads out/adaptation.npz, writes out/forcefield_reaching.mp4.
"""

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from arm import CURL, DT, L1
from figures import INK, MUTED, OUT, target_colors

W, H = 1920, 1080
FPS = 50
STRIDE = round(1 / (DT * FPS))          # simulation steps per frame (real time)
BEFORE, AFTER = 10, 120                 # steps shown around each go cue
HOLD_S = 1.5
VIEW_X, VIEW_Y = (-0.16, 0.31), (-0.04, 0.58)
STAGES = (("baseline", "1   Baseline", "no force field", False),
          ("early_field", "2   First exposure", "curl field switched on", True),
          ("late_field", "3   After learning", "same field, 150 learning updates later", True),
          ("after_effect", "4   After-effect", "field switched off", False))
BG, LIMB, LIMB_EDGE, C_FIELD = "#FAFAF9", "#E3E6EA", "#B8C0CA", "#D55E00"


def joints(q):
    elbow = L1 * np.array([np.cos(q[0]), np.sin(q[0])])
    q12 = q[0] + q[1]
    return elbow, np.array([np.cos(q12), np.sin(q12)])


class Panel:
    def __init__(self, fig, rect, data, stage, colors):
        key, title, subtitle, field = stage
        self.q, self.x, self.v = (data[f"{key}_{k}"] for k in ("q", "x", "v"))
        self.field, self.colors = field, colors
        ax = fig.add_axes(rect)
        ax.set_xlim(*VIEW_X)
        ax.set_ylim(*VIEW_Y)
        ax.set_aspect("equal")
        ax.axis("off")
        self.ax = ax
        fig.text(rect[0], rect[1] + rect[3] + 0.045, title, size=19, weight="bold", color=INK)
        fig.text(rect[0], rect[1] + rect[3] + 0.018, subtitle, size=13.5, color=C_FIELD if field else MUTED,
                 weight="semibold" if field else "normal")

        start, targets = data["start"][0], data["target"]
        # The arm sits underneath, light and translucent, so the hand paths stay readable on top of it.
        self.upper_edge, = ax.plot([], [], color=LIMB_EDGE, lw=26, solid_capstyle="round", alpha=0.7, zorder=1)
        self.fore_edge, = ax.plot([], [], color=LIMB_EDGE, lw=22, solid_capstyle="round", alpha=0.7, zorder=1)
        self.upper, = ax.plot([], [], color=LIMB, lw=23, solid_capstyle="round", zorder=2)
        self.fore, = ax.plot([], [], color=LIMB, lw=19, solid_capstyle="round", zorder=2)
        ax.add_patch(Circle((0, 0), 0.014, fc="white", ec=LIMB_EDGE, lw=2, zorder=3))           # shoulder
        self.elbow = ax.add_patch(Circle((0, 0), 0.012, fc="white", ec=LIMB_EDGE, lw=2, zorder=3))

        self.targets = [ax.add_patch(Circle(t, 0.009, fc="none", ec="#8C959F", lw=1.6, zorder=4)) for t in targets]
        ax.add_patch(Circle(start, 0.004, fc=INK, ec="none", zorder=4))
        self.done = [ax.plot([], [], color=c, lw=2.5, alpha=0.45, zorder=5)[0] for c in colors]
        self.trail, = ax.plot([], [], lw=4, solid_capstyle="round", zorder=6)
        self.hand = ax.add_patch(Circle((0, 0), 0.008, fc=INK, ec="white", lw=1.5, zorder=8))
        self.force = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=10,
                                 arrowprops=dict(arrowstyle="-|>", color=C_FIELD, lw=3, mutation_scale=22))

    def draw(self, reach, step, go, finished):
        q, x = self.q[step, reach], self.x[step, reach]
        elbow, _ = joints(q)
        self.upper.set_data([0, elbow[0]], [0, elbow[1]])
        self.upper_edge.set_data([0, elbow[0]], [0, elbow[1]])
        self.fore.set_data([elbow[0], x[0]], [elbow[1], x[1]])
        self.fore_edge.set_data([elbow[0], x[0]], [elbow[1], x[1]])
        self.elbow.center = elbow
        self.hand.center = x

        for r, target in enumerate(self.targets):
            current = r == reach and not finished
            target.set_facecolor(self.colors[r] if current and step >= go else "none")
            target.set_edgecolor(self.colors[r] if current else "#8C959F")
            target.set_linewidth(2.4 if current else 1.4)
        for r, line in enumerate(self.done):
            path = self.x[go:, r] if r < reach or finished else np.empty((0, 2))
            line.set_data(path[:, 0], path[:, 1])
        path = self.x[go:step + 1, reach] if step >= go and not finished else np.empty((0, 2))
        self.trail.set_data(path[:, 0], path[:, 1])
        self.trail.set_color(self.colors[reach])

        v = self.v[step, reach]
        f = CURL * np.array([v[1], -v[0]]) if self.field and not finished else np.zeros(2)
        visible = np.linalg.norm(f) > 0.4
        self.force.set_visible(visible)
        if visible:
            self.force.xy = x + 0.012 * f
            self.force.set_position(x)


def main():
    data = np.load(OUT / "adaptation.npz")
    go = int(data["go"][0])
    n = len(data["target"])
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.02, 0.945, "A GRU 'motor cortex' learns to reach in a curl force field", size=27, weight="bold",
             color=INK)
    fig.text(0.02, 0.905, "Top view of a two-joint arm driven by six muscles  ·  orange arrow: the field's push "
             "on the hand  ·  real time", size=15, color=MUTED)
    width, gap = 0.235, 0.012
    panels = [Panel(fig, [0.02 + i * (width + gap), 0.05, width, 0.71], data, stage, target_colors(n))
              for i, stage in enumerate(STAGES)]

    path = OUT / "forcefield_reaching.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )

    def write():
        fig.canvas.draw()
        ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())

    for reach in range(n):
        for step in range(go - BEFORE, go + AFTER, STRIDE):
            for panel in panels:
                panel.draw(reach, step, go, finished=False)
            write()
    for _ in range(round(HOLD_S * FPS)):
        for panel in panels:
            panel.draw(n - 1, go + AFTER - 1, go, finished=True)
        write()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
