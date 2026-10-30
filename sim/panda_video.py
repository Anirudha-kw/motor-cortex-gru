"""Video of the Panda reaching in the four stages of the force-field experiment, 2x2, in real time.

Each stage shows the robot (MuJoCo render of the training model: Menagerie MJX Panda without hand,
plus the handle) next to a top-down view of the handle paths, drawn like the human-arm video.
Replays out/panda_adaptation.npz; writes out/panda_forcefield_reaching.mp4.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Circle

from figures import INK, MUTED, OUT, target_colors
from panda import DT, HANDLE, PANDA_XML
from panda_cortex import CURL, REACH

W, H = 1920, 1080
FPS = 50
STRIDE = round(1 / (DT * FPS))
BEFORE, AFTER = 10, 120
HOLD_S = 1.5
ROBOT = (520, 400)                      # MuJoCo render size per stage [px]
CAMERA = dict(lookat=(0.32, 0.0, 0.42), distance=0.9, azimuth=-150.0, elevation=-24.0)
ARROW = 0.002                           # field arrow length per newton in the path view [m/N]
STAGES = (("baseline", "1   Baseline", "no force field", False),
          ("early_field", "2   First exposure", "curl field switched on", True),
          ("late_field", "3   After learning", "same field, 150 learning updates later", True),
          ("after_effect", "4   After-effect", "field switched off", False))
BG, C_FIELD = "#FAFAF9", "#D55E00"


def build_scene(home, targets, payload=False):
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for light in spec.lights:
        light.castshadow = False
    vis = spec.visual
    vis.global_.offwidth, vis.global_.offheight = ROBOT
    vis.quality.offsamples = 8
    vis.headlight.ambient = [0.45] * 3
    vis.headlight.diffuse = [0.5] * 3
    vis.headlight.specular = [0.05] * 3
    vis.rgba.haze = [0.95, 0.95, 0.96, 1.0]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.99, 0.99, 0.99], rgb2=[0.9, 0.91, 0.92], width=512, height=3072)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.88, 0.885, 0.89], rgb2=[0.85, 0.855, 0.86],
                     markrgb=[0.8, 0.81, 0.83], width=512, height=512)
    floor = spec.add_material(name="floor", texrepeat=[6, 6], texuniform=True, reflectance=0.05)
    textures = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "grid"
    floor.textures = textures
    world = spec.worldbody
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor")
    world.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0.5, 0.0, 2.5], dir=[-0.2, 0.0, -1.0],
                    castshadow=False, diffuse=[0.45] * 3, specular=[0.1] * 3)

    link7 = spec.body("link7")
    link7.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.107, 0, 0, HANDLE - 0.012], size=[0.011, 0, 0],
                   rgba=[0.22, 0.23, 0.25, 1], contype=0, conaffinity=0)
    link7.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, HANDLE], size=[0.016, 0, 0],
                   rgba=[0.12, 0.13, 0.14, 1], contype=0, conaffinity=0)
    if payload:                                                  # the payload sphere held at the handle
        link7.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, HANDLE + 0.03], size=[0.03, 0, 0],
                       rgba=[0.85, 0.62, 0.1, 1], contype=0, conaffinity=0)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=home - [0, 0, 0.017], size=[REACH + 0.04, 0.0015, 0],
                   rgba=[0.0, 0.447, 0.698, 0.12], contype=0, conaffinity=0)                 # plane of the reaches
    for target in targets:
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=target, size=[0.011, 0, 0],
                       rgba=[0.6, 0.62, 0.65, 0.6], contype=0, conaffinity=0)
    return spec.compile()


def top_down(p, home):
    """Horizontal-plane coordinates for the path view: robot's left to the left, forward up."""
    return np.stack([-(p[..., 1] - home[1]), p[..., 0] - home[0]], -1)


class Stage:
    def __init__(self, fig, rect, model, data, stage, colors, home):
        key, title, subtitle, field = stage
        self.model, self.field, self.colors, self.home = model, field, colors, home
        self.q, self.x, self.v = (data[f"{key}_{k}"] for k in ("q", "x", "v"))
        self.xy = top_down(self.x, home)
        self.data = mujoco.MjData(model)
        left, bottom, width, height = rect
        fig.text(left + 0.006, bottom + height + 0.012, title, size=16, weight="bold", color=INK)
        fig.text(left + 0.006 + 0.13, bottom + height + 0.012, subtitle, size=12.5,
                 color=C_FIELD if field else MUTED, weight="semibold" if field else "normal")

        ax = fig.add_axes([left, bottom, ROBOT[0] / W, ROBOT[1] / H])
        ax.axis("off")
        self.image = ax.imshow(np.zeros((ROBOT[1], ROBOT[0], 3), np.uint8), interpolation="none")

        size = height
        ax = fig.add_axes([left + ROBOT[0] / W + 0.01, bottom, size * H / W, size])
        r = REACH + 0.035
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal")
        ax.axis("off")
        targets = top_down(data["target"], home)
        self.targets = [ax.add_patch(Circle(t, 0.009, fc="none", ec="#8C959F", lw=1.6, zorder=2)) for t in targets]
        ax.add_patch(Circle((0, 0), 0.004, fc=INK, ec="none", zorder=2))
        self.done = [ax.plot([], [], color=c, lw=2.2, alpha=0.45, zorder=3)[0] for c in colors]
        self.trail, = ax.plot([], [], lw=3.5, solid_capstyle="round", zorder=4)
        self.hand = ax.add_patch(Circle((0, 0), 0.007, fc=INK, ec="white", lw=1.5, zorder=6))
        self.force = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=7,
                                 arrowprops=dict(arrowstyle="-|>", color=C_FIELD, lw=3, mutation_scale=20))

    def draw(self, renderer, camera, reach, step, go, finished):
        self.data.qpos[:] = self.q[step, reach]
        mujoco.mj_forward(self.model, self.data)
        renderer.update_scene(self.data, camera)
        self.image.set_data(renderer.render())

        for r, target in enumerate(self.targets):
            current = r == reach and not finished
            target.set_facecolor(self.colors[r] if current and step >= go else "none")
            target.set_edgecolor(self.colors[r] if current else "#8C959F")
            target.set_linewidth(2.4 if current else 1.6)
        for r, line in enumerate(self.done):
            path = self.xy[go:, r] if r < reach or finished else np.empty((0, 2))
            line.set_data(path[:, 0], path[:, 1])
        path = self.xy[go:step + 1, reach] if step >= go and not finished else np.empty((0, 2))
        self.trail.set_data(path[:, 0], path[:, 1])
        self.trail.set_color(self.colors[reach])
        self.hand.center = self.xy[step, reach]

        v = self.v[step, reach]
        f = CURL * np.array([v[1], -v[0]]) if self.field and not finished else np.zeros(2)
        visible = np.linalg.norm(f) > 1.0
        self.force.set_visible(visible)
        if visible:
            tip = self.xy[step, reach]
            self.force.xy = tip + ARROW * np.array([-f[1], f[0]])          # same top-down mapping as top_down()
            self.force.set_position(tip)


def main():
    data = np.load(OUT / "panda_adaptation.npz")
    go = int(data["go"][0])
    home = data["start"][0]
    colors = target_colors(len(data["target"]))
    model = build_scene(home, data["target"])
    renderer = mujoco.Renderer(model, height=ROBOT[1], width=ROBOT[0])
    camera = mujoco.MjvCamera()
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance, camera.azimuth, camera.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    fig.text(0.015, 0.955, "The same GRU 'motor cortex', now driving a 7-DoF Franka Panda", size=24, weight="bold",
             color=INK)
    fig.text(0.015, 0.922, f"Joint torques straight from the network  ·  gravity compensated  ·  curl field "
             f"{CURL:.0f} N·s/m in the horizontal plane  ·  right: handle paths from above, orange arrow = the "
             f"field's push  ·  real time", size=13.5, color=MUTED)
    stages = []
    for i, stage in enumerate(STAGES):
        col, row = i % 2, i // 2
        rect = [0.012 + col * 0.496, 0.475 - row * 0.445, 0.48, ROBOT[1] / H]
        stages.append(Stage(fig, rect, model, data, stage, colors, home))

    path = OUT / "panda_forcefield_reaching.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )

    def frame(reach, step, finished):
        for stage in stages:
            stage.draw(renderer, camera, reach, step, go, finished)
        fig.canvas.draw()
        ffmpeg.stdin.write(np.asarray(fig.canvas.buffer_rgba())[..., :3].tobytes())

    for reach in range(len(data["target"])):
        for step in range(go - BEFORE, go + AFTER, STRIDE):
            frame(reach, step, finished=False)
    for _ in range(round(HOLD_S * FPS)):
        frame(len(data["target"]) - 1, go + AFTER - 1, finished=True)
    renderer.close()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("wrote", path)


if __name__ == "__main__":
    main()
