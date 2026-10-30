#!/usr/bin/env python3
"""Operator client for the motor-cortex controller: runs blocks of episodes (baseline, curl field, washout), homing
before each episode and stopping at the first safety trip.  Run by the operator, next to the e-stop.

    rosrun panda motor_cortex_experiment.py --blocks null:2,curl:4,null:2 --n-out 6 --record ~/mc_session.npz

Each episode: set the block's field, "home" (8 s), wait for HOLD, then (unless --yes) wait for Enter, "start", wait for
"episode done" (status 10).  Any trip (status >= 20) or refusal stops the session; it never resets a trip by itself.
Telemetry rows (100 Hz) of every episode are saved with --record (columns: see motor_cortex_controller.cpp), together
with the policy log (one row per policy tick: the exact 28 observations, output u, torque) and the 1 kHz log (q, dq,
measured tau_J, tau_J_d, torque sent, gravity, tau_ext_hat_filtered) of the episode and the second after it.

Damping (--damping, default 12): the controller adds k N m s/rad of joint damping (joints 1-4; x0.2 on 5-6, x0.1 on 7)
to the policy torque.  The real arm delivers torque later and weaker than the policy's training plants; without it
the shoulder and elbow oscillate vertically and trip on joint speed (logged 2026-10-20; tests/plant_fit.py).

System identification (--pulses "4,3,100;4,-3,100"): after homing, in the stiff hold, a torque pulse of 3 N m for
100 ms on joint 4, then -3 N m; each is saved with its 1 kHz log as pulse_NN_jJ.  --blocks "" runs pulses only.

Every episode uses the SAME target order (--seed), so a curl episode can be compared reach by reach with a null one:
the null field's lateral deviation is direction-dependent (about +-2 mm), the curl's effect is a few mm on top.  The
policy adapts within an episode only -- its memory is reset at every start -- so an after-effect exists only when the
field is switched off inside an episode (--off-after K: after the K-th outward reach).
"""
import argparse
import sys
import threading
import time

import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray, Int32, String

NS = "/motor_cortex_controller"
STATUS = {0: "hold", 1: "homing", 2: "running", 3: "tripped", 10: "episode done", 11: "refused: not at home",
          12: "refused: busy", 13: "refused: too far to home", 14: "warning: handle home differs from training model",
          21: "TRIP: workspace", 22: "TRIP: joint speed", 23: "TRIP: joint limit", 24: "TRIP: non-finite policy output",
          25: "TRIP: handle speed", 15: "damping set (from the next start)"}
CURL = 13.0 * 2.72          # N s/m, the training probes' curl field


class Client:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.rows, self.policy, self.fast = [], [], []
        self.recording = False
        self.cmd = rospy.Publisher(NS + "/command", String, queue_size=10, latch=False)
        self.field = rospy.Publisher(NS + "/field", Float64MultiArray, queue_size=10, latch=True)
        rospy.Subscriber(NS + "/status", Int32, self._status)
        rospy.Subscriber(NS + "/telemetry", Float64MultiArray, self._telemetry)
        rospy.Subscriber(NS + "/policy_log", Float64MultiArray, self._policy, queue_size=1000)
        rospy.Subscriber(NS + "/fast_log", Float64MultiArray, self._fast, queue_size=5000)
        time.sleep(1.0)

    def _status(self, msg):
        with self.lock:
            self.events.append(msg.data)
        print(f"  status {msg.data}: {STATUS.get(msg.data, '?')}", flush=True)

    def _telemetry(self, msg):
        if self.recording:
            with self.lock:
                self.rows.append(list(msg.data))

    def _policy(self, msg):
        if self.recording:
            with self.lock:
                self.policy.append(list(msg.data))

    def _fast(self, msg):
        if self.recording:
            with self.lock:
                self.fast.append(list(msg.data))

    def begin_recording(self):
        with self.lock:
            self.rows, self.policy, self.fast = [], [], []
        self.recording = True

    def end_recording(self, tail=1.3):
        """Keep recording `tail` seconds (the controller logs 1 kHz for 1 s after an episode, trip or pulse)."""
        time.sleep(tail)
        self.recording = False
        with self.lock:
            return np.array(self.rows), np.array(self.policy), np.array(self.fast)

    def switch_off_at(self, reach_index, timeout=30.0):
        """Zero the field once the telemetry reports this reach index (counting outward and return reaches)."""
        start = time.time()
        while time.time() - start < timeout and not rospy.is_shutdown():
            with self.lock:
                row = self.rows[-1] if self.rows else None
            if row is not None and int(row[1]) == 2 and int(row[3]) >= reach_index:
                self.set_field(0.0)
                print(f"  field switched off at reach {reach_index}", flush=True)
                return
            time.sleep(0.005)

    def send(self, text):
        self.cmd.publish(String(data=text))

    def set_field(self, curl):
        self.field.publish(Float64MultiArray(data=[0.0, curl, -curl, 0.0, 0.0, 0.0, 0.0]))

    def request(self, text, wanted, timeout):
        """Send a command and wait for its status (events counted from before the send, so none is missed)."""
        with self.lock:
            seen = len(self.events)
        self.send(text)
        return self.wait_for(wanted, timeout, seen)

    def wait_for(self, wanted, timeout, seen=None):
        """Wait for one of the wanted status codes; return it, or the first trip/refusal, or None on timeout."""
        start = time.time()
        if seen is None:
            with self.lock:
                seen = len(self.events)
        while time.time() - start < timeout and not rospy.is_shutdown():
            with self.lock:
                new = self.events[seen:]
                seen = len(self.events)
            for code in new:
                if code in wanted or code >= 20 or code in (11, 12, 13):
                    return code
            time.sleep(0.02)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="null:2,curl:4,null:2", help="name:episodes,...; names: null, curl, anticurl")
    ap.add_argument("--n-out", type=int, default=6, help="outward reaches per episode (each followed by a return)")
    ap.add_argument("--seed", type=int, default=4, help="target order, the same for every episode (with --pairs: "
                                                      "the first pair's order; pair k uses seed + k)")
    ap.add_argument("--pairs", type=int, default=0, help="balanced design: this many null/curl pairs, each pair "
                                                         "sharing a fresh target order (overrides --blocks)")
    ap.add_argument("--off-after", type=int, default=0,
                    help="in curl episodes, switch the field off after this many outward reaches (0: never)")
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter before each episode")
    ap.add_argument("--record", default="", help="save telemetry to this .npz")
    ap.add_argument("--pulses", default="", help="joint,N m,ms;... torque pulses in hold before the blocks")
    ap.add_argument("--damping", type=float, default=12.0,
                    help="controller joint damping k [N m s/rad] on joints 1-4 (0: the policy alone, which trips)")
    args = ap.parse_args()
    rospy.init_node("motor_cortex_experiment", anonymous=True, disable_signals=True)
    c = Client()
    fields = {"null": 0.0, "curl": CURL, "anticurl": -CURL}
    plan = [(name, int(n)) for name, n in (b.split(":") for b in args.blocks.split(",") if b)]
    if args.pairs:
        # balanced design: pair k = a null and a curl episode sharing target order seed+k (so curl - null isolates
        # the field at every reach index while directions average out over pairs); which comes first alternates
        schedule = []
        for k in range(args.pairs):
            order = ("null", "curl") if k % 2 == 0 else ("curl", "null")
            schedule += [(name, args.seed + k, k, args.pairs) for name in order]
    else:
        schedule = [(name, args.seed, i, n) for name, n in plan for i in range(n)]
    if not 0.0 <= args.damping <= 30.0:
        ap.error("--damping must be 0-30 N m s/rad (the controller refuses more)")
    pulses = [tuple(float(v) for v in p.split(",")) for p in args.pulses.split(";") if p]
    for j, amp, ms in pulses:
        if not (1 <= j <= 7 and j == int(j) and abs(amp) <= 5.0 and 10.0 <= ms <= 200.0):
            ap.error(f"pulse {j},{amp},{ms}: joint 1-7, |N m| <= 5, 10-200 ms")
    episode, saved = 0, {}
    try:
        if pulses:
            c.set_field(0.0)
            print("pulses: homing", flush=True)
            c.send("home")
            code = c.wait_for({0}, 15.0)
            if code != 0:
                print(f"stopping: {STATUS.get(code, 'timeout')}", flush=True)
                return 1
            if not args.yes:
                input(f"  {len(pulses)} pulses in hold; press Enter to start (Ctrl-C to quit) ")
            for k, (j, amp, ms) in enumerate(pulses):
                time.sleep(1.0)                            # settle
                c.begin_recording()
                time.sleep(0.2)                            # 1 kHz log starts with the pulse; telemetry shows the lead-in
                c.send(f"pulse {int(j)} {amp} {ms}")
                code = c.wait_for(set(), ms / 1000.0 + 0.1)
                rows, policy, fast = c.end_recording(1.2)
                saved[f"pulse_{k:02d}_j{int(j)}"] = rows
                saved[f"pulse_{k:02d}_j{int(j)}_fast"] = fast
                peak = np.abs(fast[:, 2 + int(j) - 1] - fast[0, 2 + int(j) - 1]).max() if len(fast) else float("nan")
                print(f"  pulse {k}: joint {int(j)} {amp:+.1f} N m {ms:.0f} ms -> {len(fast)} fast rows, "
                      f"peak deflection {1e3 * peak:.2f} mrad", flush=True)
                if code is not None:
                    print(f"stopping: {STATUS.get(code, '?')}", flush=True)
                    return 1
        for name, ep_seed, i, n in schedule:
            episode += 1
            c.set_field(0.0)                       # never home inside a field
            print(f"episode {episode} ({name} {i + 1}/{n}): homing", flush=True)
            c.send("home")
            code = c.wait_for({0}, 15.0)
            if code != 0:
                print(f"stopping: {STATUS.get(code, 'timeout')}", flush=True)
                return 1
            code = c.request(f"damping {args.damping:g}", {15}, 2.0)
            if code != 15:
                print(f"stopping: damping not accepted ({STATUS.get(code, 'timeout')})", flush=True)
                return 1
            c.set_field(fields[name])
            if not args.yes:
                input(f"  field {name} ({fields[name]:+.1f} N s/m), damping {args.damping:g} set; "
                      f"press Enter to start (Ctrl-C to quit) ")
            c.begin_recording()
            c.send(f"start {args.n_out} {ep_seed}")
            if args.off_after and fields[name] != 0.0:
                threading.Thread(target=c.switch_off_at, args=(2 * args.off_after,), daemon=True).start()
            code = c.wait_for({10}, 2 * args.n_out * 0.7 + 10.0)
            c.set_field(0.0)
            rows, policy, fast = c.end_recording()
            key = f"episode_{episode:03d}_{name}" + (f"_s{ep_seed}" if args.pairs else "")
            saved[key], saved[key + "_policy"], saved[key + "_fast"] = rows, policy, fast
            print(f"  recorded {len(rows)} telemetry, {len(policy)} policy, {len(fast)} 1 kHz rows", flush=True)
            if code != 10:
                print(f"stopping: {STATUS.get(code, 'timeout')} -- inspect the arm; send 'reset' yourself",
                      flush=True)
                return 1
        print("session complete", flush=True)
        return 0
    finally:
        c.set_field(0.0)
        if args.record and saved:
            np.savez(args.record, **saved)
            n_ep = sum(k.startswith("episode_") and not k.endswith(("_policy", "_fast")) for k in saved)
            n_pulse = sum(k.startswith("pulse_") and k.endswith("_fast") for k in saved)
            print(f"telemetry saved to {args.record} ({n_ep} episodes, {n_pulse} pulses)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
