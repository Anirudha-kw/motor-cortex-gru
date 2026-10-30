"""Videos of the in-context adaptation comparison: one per condition, four controllers side by side.

Each panel shows the robot (MuJoCo) and the handle paths from above over the same episode of 6 outward
reaches and their returns, with the environment's force on the handle as an orange arrow and the
current reach's distance from the straight line (it also captures a payload's sag).  Reads
out/comparison_broad.npz (episode 0 of every condition); writes out/video_<condition>.mp4.
"""

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle

from figures import INK, MUTED, OUT
from panda_cortex import CURL, REACH
from panda_video import CAMERA, ROBOT, build_scene, top_down

W, H = 1920, 1080
FPS = 50
STRIDE = 2                               # simulation steps per frame (10 ms steps -> real time)
REACH_STEPS = 70                         # meta.REACH_STEPS
HOLD_S = 1.5
BG, C_FORCE = "#FAFAF9", "#D55E00"
PANELS = (("GRU (broad-trained)", "#D55E00"), ("GRU (velocity fields)", "#8C959F"),
          ("Impedance, stiff", "#0072B2"), ("Adaptive (Slotine-Li)", "#009E73"))
ARROW = 0.002                            # path-view arrow length per newton [m/N]
CONDITIONS = {                           # title, what the environment does, force on the handle (top view)
    "curl +35": ("Curl force field", "pushes sideways in proportion to speed (35 N·s/m)",
                 lambda x, v, home: CURL * np.array([v[1], -v[0]])),
    "push 15 N": ("Constant push", "a steady 15 N sideways push -- the old failure case",
                  lambda x, v, home: np.array([0.0, 15.0])),
    "payload 1.5 kg": ("Unknown payload", "a 1.5 kg mass at the handle the controller was not told about",
                       lambda x, v, home: np.zeros(2)),
    "spring 300 N/m (outside)": ("Spring field -- outside every controller's training and model",
                                 "pulls toward a point 5 cm to the left with 300 N/m",
                                 lambda x, v, home: -300.0 * (x[:2] - (home[:2] + np.array([0.0, 0.05])))),
}


class Panel:
    def __init__(self, fig, rect, model, arrays, cond, name, color, summary, force):
        self.model, self.force, self.color = model, force, np.array(to_rgb(color))
        self.q, self.x, self.v = (arrays[f"{cond}|{name}|traj_{k}"] for k in ("q", "x", "v"))
        self.targets, self.holds = arrays["targets"], arrays["holds"]
        self.home = self.holds[0]
        self.xy = top_down(self.x, self.home)
        self.data = mujoco.MjData(model)
        left, bottom, width, height = rect
        fig.text(left + 0.004, bottom + height + 0.014, name, size=15, weight="bold", color=color)
        dev = summary["dev_at_peak_mm"]
        fig.text(left + 0.004 + 0.16, bottom + height + 0.014,
                 f"deviation at peak speed, reach 1 → 6: {dev[0]:.1f} → {dev[-1]:.1f} mm", size=11.5, color=MUTED)

        ax = fig.add_axes([left, bottom, ROBOT[0] / W, ROBOT[1] / H])
        ax.axis("off")
        self.image = ax.imshow(np.zeros((ROBOT[1], ROBOT[0], 3), np.uint8), interpolation="none")
        ax = fig.add_axes([left + ROBOT[0] / W + 0.006, bottom, height * H / W, height])
        r = REACH + 0.04
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.axis("off")
        angles = np.arange(8) * 2 * np.pi / 8
        ring = self.home + REACH * np.stack([np.cos(angles), np.sin(angles), 0 * angles], -1)
        for t in top_down(ring, self.home):
            ax.add_patch(Circle(t, 0.008, fc="none", ec="#B8C0CA", lw=1.3, zorder=2))
        ax.add_patch(Circle((0, 0), 0.004, fc=INK, ec="none", zorder=2))
        n_reaches = len(self.x) // REACH_STEPS
        self.n_out = n_reaches // 2
        self.lines = [ax.plot([], [], lw=2.4 if r % 2 == 0 else 1.0, zorder=3,
                              color=self.shade(r // 2) if r % 2 == 0 else "#C9D1D9")[0] for r in range(n_reaches)]
        self.current_target = ax.add_patch(Circle((0, 0), 0.009, fc="none", ec=color, lw=2.4, zorder=4))
        self.hand = ax.add_patch(Circle((0, 0), 0.006, fc=INK, ec="white", lw=1.4, zorder=6))
        self.arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=7,
                                 arrowprops=dict(arrowstyle="-|>", color=C_FORCE, lw=2.6, mutation_scale=18))
        self.readout = ax.text(0, -r + 0.004, "", ha="center", va="bottom", size=11, color=INK)

    def shade(self, k):
        return tuple(1 - (0.3 + 0.7 * k / max(self.n_out - 1, 1)) * (1 - self.color))

    def draw(self, renderer, camera, step, finished):
        self.data.qpos[:] = self.q[step]
        mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data, camera)
        self.image.set_data(renderer.render())

        reach, s = divmod(step, REACH_STEPS)
        for r, line in enumerate(self.lines):
            end = (r + 1) * REACH_STEPS if (r < reach or finished) else (step + 1 if r == reach else r * REACH_STEPS)
            seg = self.xy[r * REACH_STEPS:end]
            line.set_data(seg[:, 0], seg[:, 1])
        self.current_target.center = top_down(self.targets[min(step, len(self.targets) - 1)], self.home)
        self.current_target.set_visible(not finished)
        self.hand.center = self.xy[step]

        f = self.force(self.x[step], self.v[step], self.home) if not finished else np.zeros(2)
        visible = np.linalg.norm(f) > 1.0
        self.arrow.set_visible(visible)
        if visible:
            tip = self.xy[step]
            self.arrow.xy = tip + ARROW * np.array([-f[1], f[0]])
            self.arrow.set_position(tip)

        hold, target = self.holds[reach * REACH_STEPS], self.targets[reach * REACH_STEPS]
        line = (target - hold) / np.linalg.norm(target - hold)
        rel = self.x[reach * REACH_STEPS:step + 1] - hold
        dev = np.linalg.norm(rel - (rel @ line)[:, None] * line, axis=-1).max() * 1e3
        label = "back to centre" if reach % 2 else f"reach {reach // 2 + 1} of {self.n_out}"
        self.readout.set_text("" if finished else f"{label}  ·  max deviation {dev:4.1f} mm")


def render(cond, arrays, results, out_path):
    title, description, force = CONDITIONS[cond]
    home = arrays["holds"][0]
    angles = np.arange(8) * 2 * np.pi / 8
    ring = home + REACH * np.stack([np.cos(angles), np.sin(angles), 0 * angles], -1)
    model = build_scene(home, ring, payload=cond.startswith("payload"))
    renderer = mujoco.Renderer(model, height=ROBOT[1], width=ROBOT[0])
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance, camera.azimuth, camera.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.015, 0.955, f"In-context adaptation on a 7-DoF Panda  ·  {title}", size=24, weight="bold", color=INK)
    fig.text(0.015, 0.922, f"The environment {description}.  Controllers are frozen: only the GRU's hidden state and the "
             f"adaptive controller's estimates change from reach to reach.  Real time.", size=13, color=MUTED)
    panels = []
    for i, (name, color) in enumerate(PANELS):
        col, row = i % 2, i // 2
        rect = [0.012 + col * 0.496, 0.475 - row * 0.445, 0.48, ROBOT[1] / H]
        panels.append(Panel(fig, rect, model, arrays, cond, name, color, results[cond][name], force))

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out_path)],
        stdin=subprocess.PIPE,
    )

    def frame(step, finished=False):
        for panel in panels:
            panel.draw(renderer, camera, step, finished)
        fig.canvas.draw()
        ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())

    n_steps = len(arrays["targets"])
    for step in range(0, n_steps, STRIDE):
        frame(step)
    for _ in range(round(HOLD_S * FPS)):
        frame(n_steps - 1, finished=True)
    renderer.close()
    plt.close(fig)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", out_path, flush=True)


def main():
    arrays = dict(np.load(OUT / "comparison_broad.npz"))
    results = json.loads((OUT / "comparison_broad.json").read_text())
    for cond in CONDITIONS:
        slug = cond.split(" (")[0].replace(" ", "_").replace(".", "p").replace("+", "plus").replace("/", "per")
        render(cond, arrays, results, OUT / f"video_{slug}.mp4")


if __name__ == "__main__":
    main()
