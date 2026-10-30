"""Summarise out/s2r_eval.json (s2r_eval.py): per policy, over the held-out plants and conditions -- unstable
runs, final error, lateral error at peak speed (mean and worst reach), and in-context adaptation in the curl field
(first outward reach against the mean of the last three).

    python s2r_summary.py
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
GROUPS = {"nominal": lambda p: p == "nominal", "real friction": lambda p: p == "real friction (Gaz 2019)",
          "real friction, 20 ms": lambda p: p == "real friction, 20 ms delay", "random plants": lambda p: p.startswith("random")}


def main():
    res = json.loads((OUT / "s2r_eval.json").read_text())
    summary = {}
    for pol, runs in res.items():
        summary[pol] = {}
        for gname, sel in GROUPS.items():
            rows = [(k.split(" | ")[1], v) for k, v in runs.items() if sel(k.split(" | ")[0])]
            ok = [(c, v) for c, v in rows if not v["unstable"]]
            if not rows:
                continue
            s = dict(runs=len(rows), unstable=len(rows) - len(ok))
            if ok:
                s["final_mm"] = float(np.mean([np.mean(v["final_mm"]) for _, v in ok]))
                s["lateral_mm"] = float(np.mean([np.mean(v["lateral_mm"]) for _, v in ok]))
                s["worst_lateral_mm"] = float(np.max([np.max(v["lateral_mm"]) for _, v in ok]))
                curl = [v for c, v in ok if c == "curl +35"]
                if curl:
                    s["curl_first_vs_last3_mm"] = [float(np.mean([v["lateral_mm"][0] for v in curl])),
                                                   float(np.mean([np.mean(v["lateral_mm"][-3:]) for v in curl]))]
                s["rate_limited"] = float(np.mean([v["rate_sat"] for _, v in ok]))
            summary[pol][gname] = s
    (OUT / "s2r_summary.json").write_text(json.dumps(summary, indent=1))
    for gname in GROUPS:
        print(f"\n{gname}")
        print(f"  {'policy':26s} {'unstable':>8s} {'final':>7s} {'lateral':>8s} {'worst':>7s} {'curl 1st->last3':>16s} {'rate-lim':>8s}")
        for pol, g in summary.items():
            s = g.get(gname)
            if not s:
                continue
            if "final_mm" not in s:
                print(f"  {pol:26s} {s['unstable']:>3d}/{s['runs']:<4d}")
                continue
            c = s.get("curl_first_vs_last3_mm", [np.nan, np.nan])
            print(f"  {pol:26s} {s['unstable']:>3d}/{s['runs']:<4d} {s['final_mm']:7.1f} {s['lateral_mm']:8.1f} "
                  f"{s['worst_lateral_mm']:7.1f} {c[0]:7.1f} -> {c[1]:5.1f} {s['rate_limited']:8.0%}")


if __name__ == "__main__":
    main()
