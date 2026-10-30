"""Video of what the robot should do, from record_expect's log (the real controller in the simulated loop).

Left: the arm (MuJoCo render) with the 10 cm handle, the target, the hand's recent path and the field force.
Right: the hand seen from above, magnified, with every reach's path; curl episodes are drawn over the no-field
episode's paths (same targets), so the field's push and its shrinking over the reaches are visible.

    MUJOCO_GL=egl python3 render_expect.py expect_log.txt out.mp4 [--still 1234]
"""
import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

XML = Path.home() / "mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
W, H, P = 1280, 720, 720                                   # robot view, top-view panel
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
FONT_B = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
BLUE, ORANGE, AQUA, INK, MUTED, GRID = (42, 120, 214), (235, 104, 52), (27, 175, 122), (31, 35, 40), (110, 119, 129), (226, 230, 235)
SEGMENTS = {0: ("Controller launched: the arm holds where it is", "real time"),
            1: ("home: smooth 8 s move to the trained start posture", "4x speed"),
            2: ("Episode 1: no field, 6 reaches out and back", "real time"),
            3: ("home", "4x speed"),
            4: ("Episode 2: curl field 35 N s/m", "real time"),
            5: ("home", "4x speed"),
            6: ("Episode 3: curl field, switched off after the 3rd reach", "real time")}
# log columns: t, segment, curl gain, then the telemetry row (motor_cortex_controller.cpp)
C_MODE, C_REACH, C_STEP, C_TARGET, C_HAND, C_VEL, C_Q, C_FORCE = 4, 6, 7, slice(9, 12), slice(12, 15), slice(15, 18), slice(18, 25), slice(46, 49)


def build_model():
    spec = mujoco.MjSpec.from_file(str(XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.80, 0.82, 0.85], rgb2=[0.72, 0.74, 0.78], width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.96, 0.97, 0.98], rgb2=[0.80, 0.84, 0.89], width=512, height=3072)
    mat = spec.add_material(name="floor", texrepeat=[6, 6], reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.2, 1.2, 0.05], material="floor",
                            contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0.6, -0.8, 2.2], dir=[-0.25, 0.35, -1.0], diffuse=[0.45, 0.45, 0.45],
                             specular=[0.1, 0.1, 0.1], castshadow=True)
    spec.worldbody.add_light(pos=[-0.5, 0.8, 1.8], dir=[0.3, -0.4, -1.0], diffuse=[0.2, 0.2, 0.2],
                             specular=[0.0, 0.0, 0.0], castshadow=False)
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    spec.visual.headlight.diffuse = [0.32, 0.32, 0.32]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    link7 = spec.body("link7")
    link7.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0.107, 0, 0, 0.197], size=[0.011, 0, 0],
                   rgba=[0.45, 0.47, 0.5, 1], contype=0, conaffinity=0)
    link7.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, 0.207], size=[0.016, 0, 0],
                   rgba=[0.15, 0.17, 0.2, 1], contype=0, conaffinity=0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    spec.visual.quality.shadowsize = 4096
    return spec.compile()


def add_geom(scene, gtype, size, pos, rgba, mat=None):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).flatten(), np.asarray(rgba, np.float32))
    scene.ngeom += 1


def add_segment(scene, gtype, width, a, b, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
    scene.ngeom += 1


def reach_stats(log, home):
    """Per segment, per outward reach: the hand's signed lateral position at peak speed (+ = left of the motion)."""
    out = {}
    for seg in (2, 4, 6):
        rows = log[(log[:, 1] == seg) & (log[:, C_MODE] == 2)]
        lat = []
        for r in range(0, 12, 2):
            rr = rows[rows[:, C_REACH] == r]
            if not len(rr):
                lat.append(np.nan); continue
            tgt = rr[-1, C_TARGET]                      # the first row can still show the previous target
            d = (tgt - home)[:2] / np.linalg.norm((tgt - home)[:2])
            n = np.array([-d[1], d[0]])
            k = int(np.argmax(np.linalg.norm(rr[:, C_VEL][:, :2], axis=1)))
            lat.append(float((rr[k, C_HAND] - home)[:2] @ n))
        out[seg] = np.array(lat)
    return out


class TopView:
    def __init__(self, log, home):
        self.home, self.scale, self.half = home, P / 0.18, 0.09                # +-9 cm
        self.font = ImageFont.truetype(FONT, 17)
        self.small = ImageFont.truetype(FONT, 13)
        self.bold = ImageFont.truetype(FONT_B, 19)
        self.log = log
        self.targets = np.unique(np.round(log[(log[:, C_MODE] == 2), C_TARGET], 5), axis=0)
        self.null_paths = self.paths(2, len(log))
        self.stats = reach_stats(log, home)

    def px(self, p):
        """World (x forward, y left) -> panel pixels: forward is up, left is left."""
        rel = p[:2] - self.home[:2]
        return (P / 2 - rel[1] * self.scale, P / 2 - rel[0] * self.scale + 30)

    def paths(self, seg, upto):
        rows = self.log[:upto]
        rows = rows[(rows[:, 1] == seg) & (rows[:, C_MODE] == 2)]
        out = []
        for r in range(12):
            rr = rows[rows[:, C_REACH] == r]
            if len(rr) > 1:
                out.append((r, rr[:, C_HAND], rr[0, 2] != 0.0 or (seg == 6 and r < 6)))
        return out

    def draw(self, i):
        row = self.log[i]
        seg = int(row[1])
        img = Image.new("RGB", (P, H), (248, 249, 251))
        d = ImageDraw.Draw(img)
        for k in range(-9, 10):                                                  # 1 cm grid
            x0, y0 = self.px(self.home + np.array([k * 0.01, -0.09, 0]))
            x1, y1 = self.px(self.home + np.array([k * 0.01, 0.09, 0]))
            d.line([(x0, y0), (x1, y1)], fill=GRID, width=1)
            x0, y0 = self.px(self.home + np.array([-0.09, k * 0.01, 0]))
            x1, y1 = self.px(self.home + np.array([0.09, k * 0.01, 0]))
            d.line([(x0, y0), (x1, y1)], fill=GRID, width=1)
        for t in self.targets:
            x, y = self.px(t)
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(190, 196, 204), width=2)
        if seg in (4, 6):                                                        # no-field paths, for comparison
            for _, path, _ in self.null_paths:
                d.line([self.px(p) for p in path], fill=(196, 202, 210), width=3)
        if seg in (2, 4, 6):
            for r, path, field_on in self.paths(seg, i + 1):
                colour = BLUE if seg == 2 else (ORANGE if field_on else AQUA)
                d.line([self.px(p) for p in path], fill=colour, width=3)
            if int(row[C_MODE]) == 2:
                x, y = self.px(row[C_TARGET])
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(42, 120, 214, 90), outline=BLUE, width=3)
        x, y = self.px(row[C_HAND])
        f = row[C_FORCE]
        if np.linalg.norm(f) > 0.5:
            tip = (x - f[1] * 0.0015 * self.scale, y - f[0] * 0.0015 * self.scale)   # 1.5 mm per N
            d.line([(x, y), tip], fill=(214, 60, 50), width=4)
            d.ellipse([tip[0] - 4, tip[1] - 4, tip[0] + 4, tip[1] + 4], fill=(214, 60, 50))
        d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=INK)
        d.text((24, 16), "Hand from above (forward is up)", font=self.bold, fill=INK)
        sb = (P - 24 - 0.01 * self.scale, H - 92)                             # 1 cm scale bar, bottom right
        d.line([sb, (sb[0] + 0.01 * self.scale, sb[1])], fill=INK, width=3)
        d.text((sb[0], sb[1] + 6), "1 cm", font=self.small, fill=MUTED)
        legend = [(BLUE, "no field"), (ORANGE, "curl field on"), (AQUA, "field switched off"),
                  ((196, 202, 210), "no-field paths, same targets"), ((214, 60, 50), "field force, 1.5 mm per N")]
        for k, (c, label) in enumerate(legend):
            yy = 46 + 19 * k
            d.line([(24, yy + 9), (48, yy + 9)], fill=c, width=4)
            d.text((56, yy), label, font=self.small, fill=MUTED)
        if seg in (4, 6):                                                        # push vs no field, per reach done
            done = int(row[C_REACH]) // 2 if int(row[C_MODE]) == 2 else 6
            vals = [1e3 * (self.stats[seg][r] - self.stats[2][r]) for r in range(min(done, 6))]
            txt = "  ".join(f"{-v:+.1f}" for v in vals)
            d.text((24, H - 62), "sideways push at peak speed vs no field (mm to the right):", font=self.small, fill=MUTED)
            d.text((24, H - 38), txt if txt else "—", font=self.font, fill=INK)
        return img


def banner(frame, row, title, speed):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 78], fill=(255, 255, 255, 215))
    d.text((24, 12), title, font=ImageFont.truetype(FONT_B, 24), fill=INK)
    mode = {0: "HOLD", 1: "HOMING", 2: "RUNNING", 3: "TRIPPED"}.get(int(row[C_MODE]), "?")
    sub = f"{mode}   ·   {speed}"
    if int(row[C_MODE]) == 2:
        r = int(row[C_REACH])
        sub += f"   ·   reach {r // 2 + 1} of 6 {'out' if r % 2 == 0 else 'back'}"
        sub += f"   ·   {'curl field ON' if row[2] != 0 else 'no field'}"
    d.text((24, 46), sub, font=ImageFont.truetype(FONT, 17), fill=MUTED)
    d.text((W - 360, H - 34), "simulated Panda · the controller's own code", font=ImageFont.truetype(FONT, 15),
           fill=MUTED)
    return img


def card(lines, size=(W + P, H)):
    img = Image.new("RGB", size, (248, 249, 251))
    d = ImageDraw.Draw(img)
    y = 230
    for k, line in enumerate(lines):
        f = ImageFont.truetype(FONT_B if k == 0 else FONT, 36 if k == 0 else 22)
        w = d.textlength(line, font=f)
        d.text(((size[0] - w) / 2, y), line, font=f, fill=INK if k == 0 else MUTED)
        y += 64 if k == 0 else 38
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--still", type=int, default=None, help="render one row to a PNG instead")
    args = ap.parse_args()
    log = np.loadtxt(args.log)
    run = log[log[:, C_MODE] == 2]
    back = run[run[:, C_REACH] == 1]
    home = back[-1, C_TARGET]                                            # return reaches target home (last row: fresh)
    model = build_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.33, 0.0, 0.33]
    cam.distance, cam.azimuth, cam.elevation = 1.05, 140.0, -20.0
    top = TopView(log, home)

    def frame(i):
        row = log[i]
        data.qpos[:7] = row[C_Q]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        sc = renderer.scene
        seg = int(row[1])
        if int(row[C_MODE]) == 2:
            add_geom(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.011, 0, 0], row[C_TARGET], [0.16, 0.47, 0.84, 0.55])
        start = max(0, i - 60)
        pts = [log[k, C_HAND] for k in range(start, i + 1) if int(log[k, 1]) == seg]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm(b - a) > 1e-5:
                add_segment(sc, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, a, b, [0.92, 0.41, 0.2, 0.9])
        f = row[C_FORCE]
        if np.linalg.norm(f) > 0.5:
            add_segment(sc, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, row[C_HAND], row[C_HAND] + 0.006 * f,
                        [0.84, 0.24, 0.2, 1.0])
        img = renderer.render()
        title, speed = SEGMENTS[seg]
        left = banner(img, row, title, speed)
        both = Image.new("RGB", (W + P, H))
        both.paste(left, (0, 0))
        both.paste(top.draw(i), (W, 0))
        return np.asarray(both)

    if args.still is not None:
        Image.fromarray(frame(args.still)).save(args.out)
        print("wrote", args.out)
        return
    fps = 50
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W + P}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    s = reach_stats(log, home)
    push = [1e3 * (s[4][r] - s[2][r]) for r in range(6)]
    intro = card(["Motor-cortex controller on the Panda: expected behaviour",
                  "Simulated arm with real-Panda joint friction, driven by the controller's own C++ code",
                  "First-session settings: 6 cm reaches, torque caps 20 / 5 N m",
                  "Left: the arm.   Right: the hand seen from above, magnified"])
    for _ in range(int(3.5 * fps)):
        enc.stdin.write(intro.tobytes())
    i, n = 0, 0
    while i < len(log):
        enc.stdin.write(frame(i).tobytes())
        n += 1
        i += 8 if int(log[i, 1]) in (1, 3, 5) else 2               # homing at 4x, everything else real time
    outro = card(["What to look for on the robot",
                  "Reaches: 6 cm, peak speed about 0.37 m/s, about 0.2 s after the target appears",
                  "Curl field pushes the hand right: " + ", ".join(f"{-v:.1f}" for v in push) + " mm over the 6 reaches",
                  "The network's memory resets every episode, so each curl episode starts naive",
                  "Switching the field off mid-episode leaves a tiny after-effect (~0.3 mm)"])
    for _ in range(int(5 * fps)):
        enc.stdin.write(outro.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"wrote {args.out}: {n} frames of motion, {len(log) / 100:.1f} s recorded")


if __name__ == "__main__":
    main()
