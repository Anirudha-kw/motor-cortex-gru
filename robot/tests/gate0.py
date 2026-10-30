"""Gate 0 (GATE0.md): fixed damping and hand-set schedules on the plants fitted to the real arm, through the real
controller (expect.cpp).  Scores stability, adaptation ratio, after-effect and effort, and applies the decision.

    python3 gate0.py <expect binary> [out.json] [weights.mcgru]
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights/motor_cortex_s2rR_s1.mcgru"
SCRATCH = Path(os.environ.get("GATE0_SCRATCH", "/tmp/gate0"))
FIXED = [0, 4, 8, 12, 16, 20, 25, 30]
SCHEDULES = ["decay", "within", "within_rev"]
REAL_UNSTABLE = {12}           # swung 23 mm on the real arm (limit 10): unstable whatever simulation says


def plants():
    out = {}
    for i, r in enumerate(np.load(HERE / "plant_fit2_top.npy", allow_pickle=True)[:6]):
        mjb = SCRATCH / f"gate0_fit{i}.mjb"
        if not mjb.exists():
            env = dict(os.environ, ARMATURE=f"{r['armature']:.5f}", TOOL="0.63", FRICTION_SCALE="0", DAMPING="0")
            subprocess.run([sys.executable, str(HERE / "make_harness_model.py"), str(mjb)], env=env, check=True,
                           capture_output=True)
        f = lambda a: ",".join(f"{v:.5f}" for v in a)
        out[f"fit{i}"] = (mjb, dict(PLANT_GAIN=f(r["gain"]), PLANT_LAG=f(r["lag"]), PLANT_FRIC=f(r["friction"])))
    out["nominal"] = (HERE / "harness_panda.mjb", {})
    return out


def parse(text):
    eps, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"SUMMARY (\w) status (\d+) rows (\d+) maxz_mm ([\d.]+) dqmax ([\d.]+) effort ([\d.]+)", line)
        if m:
            cur = m.group(1)
            eps[cur] = dict(status=int(m.group(2)), rows=int(m.group(3)), maxz=float(m.group(4)),
                            dqmax=float(m.group(5)), effort=float(m.group(6)), d=[])
            continue
        m = re.search(r"minus null ([+-][\d.]+)", line)
        if m and cur:
            eps[cur]["d"].append(float(m.group(1)))
    return eps


def score(eps):
    stable = all(eps.get(k, {}).get("status") == 10 and eps[k]["maxz"] < 10.0 for k in "ABC")
    d = eps.get("B", {}).get("d", [])
    if len(d) < 6:
        return dict(stable=False, A=float("nan"), after=float("nan"), after_ok=False, effort=float("nan"))
    A = abs(d[0]) / max(np.mean(np.abs(d[4:6])), 1e-6)
    c = eps.get("C", {}).get("d", [])
    after = c[3] if len(c) >= 4 else float("nan")
    return dict(stable=stable, A=A, after=after, after_ok=bool(np.sign(after) == -np.sign(d[0])),
                effort=float(np.mean([eps[k]["effort"] for k in "ABC" if k in eps])),
                maxz=max(eps[k]["maxz"] for k in "ABC" if k in eps), d=d)


def run(binary, mjb, env, setting, weights=WEIGHTS):
    e = dict(os.environ, **env)
    e.pop("DAMPING_K", None)
    e.pop("SCHEDULE", None)
    if isinstance(setting, str):
        e["SCHEDULE"] = setting
        e["DAMPING_K"] = "20"           # the start value; the schedule overwrites it every tick
    else:
        e["DAMPING_K"] = str(setting)
    out = subprocess.run([binary, str(mjb), str(weights)], env=e, capture_output=True, text=True, timeout=600)
    return score(parse(out.stdout))


def main():
    binary = sys.argv[1]
    weights = Path(sys.argv[3]) if len(sys.argv) > 3 else WEIGHTS
    print(f"policy {weights.name}")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    pl = plants()
    jobs = [(p, s) for p in pl for s in FIXED + SCHEDULES]
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda j: run(binary, *pl[j[0]], j[1], weights), jobs)))
    fits = [p for p in pl if p.startswith("fit")]
    print(f"{'setting':11s} | " + " | ".join(f"{p:^22s}" for p in pl) + " | verdict on fitted plants")
    print(f"{'':11s} | " + " | ".join(f"{'stab  A   after  eff':^22s}" for _ in pl) + " |")
    table = {}
    for s in FIXED + SCHEDULES:
        cells = []
        for p in pl:
            r = res[(p, s)]
            cells.append(f"{'ok ' if r['stable'] else 'NO '} {r['A']:4.1f} {r['after']:+5.2f}{'*' if r['after_ok'] else ' '} "
                         f"{r['effort']:4.2f}")
        stable_all = all(res[(p, s)]["stable"] for p in fits) and s not in REAL_UNSTABLE
        a_min = min(res[(p, s)]["A"] for p in fits)
        passes = stable_all and a_min >= 2.0
        table[str(s)] = dict(stable_all=stable_all, A_min=a_min, passes=passes,
                             per_plant={p: {k: v for k, v in res[(p, s)].items()} for p in pl})
        tag = ("PASS" if passes else "fail") + f" (stable on all: {stable_all}{', real 12 unstable' if s in REAL_UNSTABLE else ''}; min A {a_min:.1f})"
        print(f"{str(s):11s} | " + " | ".join(f"{c:22s}" for c in cells) + f" | {tag}")
    fixed_pass = [s for s in FIXED if table[str(s)]["passes"]]
    sched_pass = [s for s in SCHEDULES if table[s]["passes"]]
    if fixed_pass:
        verdict = f"NO HEADROOM: fixed damping {fixed_pass} passes -- deploy it, drop learned impedance"
    elif sched_pass:
        verdict = f"HEADROOM: no fixed damping passes, schedule(s) {sched_pass} do -- proceed with learned impedance"
    else:
        verdict = ("NEITHER: no fixed damping and no hand-set schedule passes -- impedance scheduling alone does not "
                   "resolve the trade-off for this policy; learned impedance survives only as a retraining question")
    print("\n" + verdict)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(dict(table=table, verdict=verdict), indent=1, default=float))


if __name__ == "__main__":
    main()
