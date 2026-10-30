"""Where is the jitter? Band RMS of dq, policy torque and damping torque on the running segment of 1 kHz fast logs.

    python3 jitter.py ../../sim/out/mc_d20_session.npz:20 ../../sim/out/mc_damped_1.npz:12     (file:damping k)
"""
import sys
import numpy as np

SHAPE = np.array([1, 1, 1, 1, 0.2, 0.2, 0.1])
BANDS = [(0, 3), (3, 8), (8, 15), (15, 40), (40, 95), (95, 105), (105, 500)]


def bands(x, fs=1000.0):
    x = x - x.mean(0)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    tot = X.sum(0)
    return np.array([X[(f >= a) & (f < b)].sum(0) / tot for a, b in BANDS]), f, X


def rms_band(x, lo, hi, fs=1000.0):
    X = np.fft.rfft(x - x.mean(0), axis=0)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < lo) | (f >= hi)] = 0
    return np.fft.irfft(X, n=len(x), axis=0).std(0)


for path, k in [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[1:]]:
    z = np.load(path)
    for key in [x for x in z.files if x.endswith("_fast")]:
        F = z[key]
        run = F[F[:, 1] == 2]
        if len(run) < 2000:
            continue
        t = run[:, 0]
        gaps = np.diff(t)
        dq, tau = run[:, 9:16], run[:, 30:37]
        damp = -k * SHAPE * dq
        pol = tau - damp
        print(f"\n{path.split('/')[-1]} {key}  k={k:g}  {len(run)} rows, dt median {1e3*np.median(gaps):.2f} ms, "
              f"max gap {1e3*gaps.max():.1f} ms")
        print("  RMS by band, joints 1-4         " + "  ".join(f"{a}-{b}Hz".rjust(9) for a, b in BANDS[1:5]))
        for name, x, u in [("dq (mrad/s)", dq, 1e3), ("policy tau (Nm)", pol, 1), ("damping tau (Nm)", damp, 1),
                           ("tau sent (Nm)", tau, 1), ("tau_J meas (Nm)", run[:, 16:23] - run[:, 37:44], 1)]:
            for j in range(4):
                r = [u * rms_band(x[:, j:j+1], a, b)[0] for a, b in BANDS[1:5]]
                print(f"  {name:17s} j{j+1}  " + "  ".join(f"{v:9.3f}" for v in r))
        # policy torque steps at the 10 ms ticks
        d = np.abs(np.diff(pol[:, :4], axis=0))
        print("  policy tau |step| per ms: median %s  p99 %s  max %s" % (
            np.round(np.median(d, 0), 3), np.round(np.percentile(d, 99, 0), 3), np.round(d.max(0), 2)))
        sat = np.abs(np.diff(tau[:, :4], axis=0)) > 0.99
        print("  rate-limit active (|dtau|>0.99 Nm/ms) fraction per joint:", np.round(sat.mean(0), 4))
        # dominant peak of dq in 3-40 Hz
        _, f, X = bands(dq[:, :4])
        m = (f > 3) & (f < 40)
        print("  dq peak frequency 3-40 Hz per joint:", np.round(f[m][X[m].argmax(0)], 1))
