"""Does the GRU's curl adaptation come from stiffening (impedance) or from compensating the field (internal model)?

A lateral probe (±F for 50 ms at the hand, from step 16 of one outward reach) is applied in the null and the curl
episode of the same target order, through the real controller (expect.cpp, PROBE_REACH / PROBE_N).  Response =
half the difference of the +F and -F lateral traces, which cancels the unperturbed path.  If the arm stiffens as it
adapts, the curl response falls from reach 1 to reach 6 relative to the null response (difference in differences).
The run also reports the sim's curl-null deviation (the decline being explained) and its after-effect (episode C).

    python3 probe_impedance.py <expect binary> <scratch dir with fit{0,1,2}.mjb> [out.json]
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from gate0 import WEIGHTS, parse

HERE = Path(__file__).resolve().parent
SEEDS = range(1, 17)
REACHES = range(6)
F = 5.0
K = "20"


def plants(scratch):
    top = np.load(HERE / "plant_fit2_top.npy", allow_pickle=True)[:3]
    f = lambda a: ",".join(f"{v:.5f}" for v in a)
    out = {"nominal": (HERE / "harness_panda.mjb", {})}
    for i, r in enumerate(top):
        out[f"fit{i}"] = (Path(scratch) / f"fit{i}.mjb",
                          dict(PLANT_GAIN=f(r["gain"]), PLANT_LAG=f(r["lag"]), PLANT_FRIC=f(r["friction"])))
    return out


def run(binary, mjb, env, seed, reach, force):
    e = dict(os.environ, **env, DAMPING_K=K, SEED=str(seed), PROBE_REACH=str(reach), PROBE_N=str(force))
    out = subprocess.run([binary, str(mjb), str(WEIGHTS)], env=e, capture_output=True, text=True, timeout=900).stdout
    tr = {l.split()[1]: np.array([float(x) for x in l.split()[2:]]) for l in out.splitlines()
          if l.startswith("PTRACE")}
    eps = parse(out)
    ok = all(eps.get(x, {}).get("status") == 10 for x in "AB")
    return ok, tr, eps


def main():
    binary, scratch = sys.argv[1], sys.argv[2]
    pl = plants(scratch)
    jobs = [(p, s, r, sg) for p in pl for s in SEEDS for r in REACHES for sg in (1, -1)]
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda j: run(binary, *pl[j[0]], j[1], j[2], j[3] * F), jobs)))
    table = {}
    print(f"damping {K}, probe ±{F:g} N x 50 ms; response = peak of (+F - -F)/2 over 400 ms, mm; mean over orders")
    for p in pl:
        resp = {"A": np.full((len(SEEDS), 6), np.nan), "B": np.full((len(SEEDS), 6), np.nan)}
        for i, s in enumerate(SEEDS):
            for r in REACHES:
                (ok1, t1, _), (ok2, t2, _) = res[(p, s, r, 1)], res[(p, s, r, -1)]
                if not (ok1 and ok2):
                    continue
                for ep in "AB":
                    n = min(len(t1.get(ep, [])), len(t2.get(ep, [])))
                    if n >= 20:
                        resp[ep][i, r] = np.max((t1[ep][:n] - t2[ep][:n]) / 2)
        # the sim's own curl - null deviation and after-effect, from the probe-free reaches of the reach-5 runs
        d = np.array([[x for x in res[(p, s, 5, 1)][2].get("B", {}).get("d", [])][:5] for s in SEEDS
                      if res[(p, s, 5, 1)][0] and len(res[(p, s, 5, 1)][2].get("B", {}).get("d", [])) >= 5])
        after = np.nanmean([res[(p, s, 0, 1)][2].get("C", {}).get("d", [np.nan] * 4)[3] for s in SEEDS])
        a, b = np.nanmean(resp["A"], 0), np.nanmean(resp["B"], 0)
        ratio = b / a
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(2000):
            idx = rng.integers(0, len(SEEDS), len(SEEDS))
            ra = np.nanmean(resp["B"][idx], 0) / np.nanmean(resp["A"][idx], 0)
            boot.append(ra[0] - ra[5])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"\n{p}:  stable {np.mean([res[(p, s, r, sg)][0] for s in SEEDS for r in REACHES for sg in (1, -1)]):.0%}")
        print("   reach                 " + "  ".join(f"{r + 1:6d}" for r in REACHES))
        print("   null response (mm)    " + "  ".join(f"{v:6.3f}" for v in a))
        print("   curl response (mm)    " + "  ".join(f"{v:6.3f}" for v in b))
        print("   curl / null           " + "  ".join(f"{v:6.3f}" for v in ratio))
        print(f"   stiffening = (curl/null) reach 1 - reach 6 = {ratio[0] - ratio[5]:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        if len(d):
            print("   curl - null deviation, reaches 1-5 (probe-free): " + " ".join(f"{v:+.2f}" for v in d.mean(0))
                  + f";  after-effect (field off, reach 4): {after:+.2f} mm")
        table[p] = dict(null=a.tolist(), curl=b.tolist(), ratio=ratio.tolist(), stiffening=[ratio[0] - ratio[5], lo, hi],
                        after=after)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
