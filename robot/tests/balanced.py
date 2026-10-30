"""Gate 0 again, with a design that separates adaptation from reach direction.

Gate 0 and the first hardware session used one target order (seed 4), so "reach 1 vs reach 6" also compared
directions.  Here every (plant, damping) runs 16 target orders; for each order the curl and null episodes share the
order, so curl - null (signed lateral deviation at peak speed, perpendicular to the movement) isolates the field's
effect at each reach index, and averaging over orders balances directions.  Through the real controller
(expect.cpp), 6 cm reaches, first-session caps.

    python3 balanced.py <expect binary> [out.json] [weights.mcgru]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from gate0 import WEIGHTS, parse, plants

SEEDS = range(1, 17)
DAMPING = [0, 8, 12, 16, 20]


def run(binary, mjb, env, k, seed, weights):
    e = dict(os.environ, **env, DAMPING_K=str(k), SEED=str(seed))
    e.pop("SCHEDULE", None)
    out = subprocess.run([binary, str(mjb), str(weights)], env=e, capture_output=True, text=True, timeout=600)
    eps = parse(out.stdout)
    ok = all(eps.get(x, {}).get("status") == 10 and eps[x]["maxz"] < 10.0 for x in "ABC")
    d = eps.get("B", {}).get("d", [])
    c = eps.get("C", {}).get("d", [])
    return dict(stable=ok, d=d if len(d) == 6 else None, after=c[3] if len(c) >= 4 else None)


def main():
    binary = sys.argv[1]
    weights = Path(sys.argv[3]) if len(sys.argv) > 3 else WEIGHTS
    pl = plants()
    jobs = [(p, k, s) for p in pl for k in DAMPING for s in SEEDS]
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda j: run(binary, *pl[j[0]], j[1], j[2], weights), jobs)))
    table = {}
    print(f"policy {weights.name}; 16 target orders per cell; curl - null at peak speed, mean over orders (mm)")
    print(f"{'plant':8s} {'k':>3s} {'stable':>7s}   reach 1 .. 6                                  ratio r1/r56   after-effect")
    for p in pl:
        for k in DAMPING:
            rows = [res[(p, k, s)] for s in SEEDS]
            stable = np.mean([r["stable"] for r in rows])
            ds = np.array([r["d"] for r in rows if r["stable"] and r["d"] is not None])
            if len(ds) < 4:
                print(f"{p:8s} {k:3d} {stable:7.0%}   (too few stable runs)")
                continue
            m = ds.mean(0)
            se = ds.std(0) / np.sqrt(len(ds))
            ratio = m[0] / np.mean(m[4:6]) if abs(np.mean(m[4:6])) > 1e-6 else float("inf")
            after = np.mean([r["after"] for r in rows if r["stable"] and r["after"] is not None])
            table[f"{p}/{k}"] = dict(stable=float(stable), mean=m.tolist(), se=se.tolist(), ratio=float(ratio),
                                     after=float(after), n=len(ds))
            print(f"{p:8s} {k:3d} {stable:7.0%}   " + " ".join(f"{v:+5.2f}" for v in m)
                  + f"   (se ~{se.mean():.2f})   {ratio:5.2f}   {after:+.2f}")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
