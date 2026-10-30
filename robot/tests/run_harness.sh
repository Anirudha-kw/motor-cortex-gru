#!/bin/bash
# Build and run the closed-loop harness and the core test locally (no ROS needed).
#   MUJOCO_PKG=<pip mujoco package dir> bash tests/run_harness.sh
set -eu
cd "$(dirname "$0")"
OUT=${OUT:-/tmp/motor_cortex_tests}; mkdir -p "$OUT"
g++ -O2 -std=c++14 -Wall -Wextra -I/usr/include/eigen3 -I../include test_core.cpp -o "$OUT/test_core"
"$OUT/test_core" ../weights/motor_cortex_s2rR_s1.mcgru trace_s2rR_s1.txt
[ -f harness_panda.mjb ] || python3 make_harness_model.py
g++ -O2 -std=c++17 -Wall -I/usr/include/eigen3 -Iharness_stubs -I../include -I"$MUJOCO_PKG/include" harness.cpp \
  -o "$OUT/harness" "$MUJOCO_PKG/libmujoco.so."* -Wl,-rpath,"$MUJOCO_PKG"
"$OUT/harness" harness_panda.mjb ../weights/motor_cortex_s2rR_s1.mcgru

# The plant fitted to the real reach (fit 0): the controller must hold it with "damping 12"
[ -f harness_fit0.mjb ] || ARMATURE=0.031 TOOL=0.63 FRICTION_SCALE=0 DAMPING=0 python3 make_harness_model.py harness_fit0.mjb
FIRST_SESSION=1 REAL_FIT=0 DAMPING_K=12 "$OUT/harness" harness_fit0.mjb ../weights/motor_cortex_s2rR_s1.mcgru
