"""Prediction for a catch-trial session (motor_cortex_experiment.py --pairs N --seed S --off-after 5): the curl - null
deviation on outward reaches 1-5 (field on) and the after-effect on reach 6 (field off in the curl episode), through
the real controller (expect.cpp, OFF_AFTER=5), on the exact target orders the robot will use.

    python3 catch_predict.py <expect binary> <scratch dir with fit{0,1,2}.mjb> <first seed> <pairs>
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gate0 import WEIGHTS, parse
from probe_impedance import plants


def run(binary, mjb, env, seed):
    e = dict(os.environ, **env, DAMPING_K="20", SEED=str(seed), OFF_AFTER="5")
    eps = parse(subprocess.run([binary, str(mjb), str(WEIGHTS)], env=e, capture_output=True, text=True,
                               timeout=900).stdout)
    if not all(eps.get(x, {}).get("status") == 10 and len(eps[x]["d"]) == 6 for x in "ABC"):
        return None
    return np.array(eps["B"]["d"]), np.array(eps["C"]["d"])        # "minus null" columns, mm


def main():
    binary, scratch, s0, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    pl = plants(scratch)
    jobs = [(p, s) for p in pl for s in range(s0, s0 + n)]
    with ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda j: run(binary, *pl[j[0]], j[1]), jobs)))
    print(f"seeds {s0}-{s0 + n - 1}, damping 20; catch episode = curl with the field off on outward reach 6")
    print("plant     stable   curl-null reaches 1-5 (field on)           reach 6 field on | AFTER-EFFECT (off)   decline d1-d5")
    for p in pl:
        ok = [res[(p, s)] for s in range(s0, s0 + n) if res[(p, s)] is not None]
        B = np.array([b for b, _ in ok]); C = np.array([c for _, c in ok])
        dec = abs(C[:, 0].mean()) - abs(C[:, 4].mean())
        print(f"{p:8s} {len(ok)}/{n}   " + " ".join(f"{v:+5.2f}" for v in C[:, :5].mean(0))
              + f"      {B[:, 5].mean():+5.2f}        |   {C[:, 5].mean():+5.2f} (sd {C[:, 5].std():.2f})        {dec:+.2f}")


if __name__ == "__main__":
    main()
