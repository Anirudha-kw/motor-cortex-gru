#!/usr/bin/env bash
# Install the motor-cortex controller into the robot host's `panda` package.  Run manually on the robot host.
# It never starts or stops ROS nodes and never moves the robot.
#
#   bash install_robot.sh            # default: build a COPY of the package in an isolated /tmp workspace and stop
#   bash install_robot.sh install    # back up, add the controller to ~/catkin_ws/src/panda, rebuild the workspace
#
# Adds 5 files + the policy weights, and two additive edits: one source line in CMakeLists.txt's add_library
# (after src/variable_impedance_controller.cpp) and one <class> in panda_controllers_plugin.xml.  Existing
# controllers' sources are not touched.
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:-isolated}
PKG=${PKG:-$HOME/catkin_ws/src/panda}
FRANKA_DIR=${FRANKA_DIR:-$HOME/libfranka/build}
cd "$HERE"
sha256sum --check --quiet bundle.sha256
test -d "$PKG" || { echo "no package at $PKG" >&2; exit 1; }
grep -q "src/variable_impedance_controller.cpp" "$PKG/CMakeLists.txt" || { echo "CMakeLists.txt anchor not found" >&2; exit 1; }
grep -q "</library>" "$PKG/panda_controllers_plugin.xml" || { echo "plugin xml anchor not found" >&2; exit 1; }
for f in include/panda/motor_cortex_core.h include/panda/motor_cortex_controller.h src/motor_cortex_controller.cpp \
         config/motor_cortex_controller.yaml config/motor_cortex_first_session.yaml launch/motor_cortex_controller.launch; do
  test ! -e "$PKG/$f" || { echo "$PKG/$f already exists: refusing to overwrite" >&2; exit 1; }
done

add_to() {   # $1: a package directory to modify
  local P=$1
  mkdir -p "$P/weights" "$P/scripts"
  install -m 644 include/panda/motor_cortex_core.h include/panda/motor_cortex_controller.h "$P/include/panda/"
  install -m 644 src/motor_cortex_controller.cpp "$P/src/"
  install -m 644 config/motor_cortex_controller.yaml config/motor_cortex_first_session.yaml "$P/config/"
  install -m 644 launch/motor_cortex_controller.launch "$P/launch/"
  install -m 644 weights/*.mcgru "$P/weights/"
  install -m 755 scripts/motor_cortex_experiment.py "$P/scripts/"
  python3 - "$P" <<'PY'
import sys, re
p = sys.argv[1]
cm = open(p + "/CMakeLists.txt").read()
m = re.search(r"add_library\(panda_controllers[^)]*\)", cm)
assert m and "src/variable_impedance_controller.cpp" in m.group(0), "add_library(panda_controllers ...) not as expected"
block = m.group(0).replace("src/variable_impedance_controller.cpp",
                           "src/variable_impedance_controller.cpp\n  src/motor_cortex_controller.cpp", 1)
open(p + "/CMakeLists.txt", "w").write(cm.replace(m.group(0), block, 1))
px = open(p + "/panda_controllers_plugin.xml").read()
cls = """  <class name="panda_controllers/MotorCortexController"
         type="panda_controllers::MotorCortexController"
         base_class_type="controller_interface::ControllerBase">
    <description>
      In-context GRU policy (motor-cortex-gru, sim-to-real retrained) driving joint torques at 100 Hz inside the 1 kHz
      loop, with a virtual force field at the handle and safety trips.
    </description>
  </class>
</library>"""
open(p + "/panda_controllers_plugin.xml", "w").write(px.replace("</library>", cls, 1))
print("edited CMakeLists.txt and panda_controllers_plugin.xml in", p)
PY
}

build() {    # $1: workspace
  unset PYTHONPATH
  source /opt/ros/noetic/setup.bash
  source "$HOME/catkin_ws/devel/setup.bash"
  catkin_make -C "$1" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DFranka_DIR="$FRANKA_DIR" -j2 -l2
}

if [ "$MODE" = isolated ]; then
  WS=$(mktemp -d /tmp/motor_cortex_ws.XXXXXX)
  mkdir -p "$WS/src"
  cp -a "$PKG" "$WS/src/panda"
  add_to "$WS/src/panda"
  build "$WS"
  LIB="$WS/devel/lib/libpanda_controllers.so"
  if [ "$(nm -C -D "$LIB" | grep -c "MotorCortexController::update")" = 0 ]; then   # grep -q + pipefail = SIGPIPE
    echo "Build finished but $LIB does not contain MotorCortexController -- stop and report this." >&2
    exit 1
  fi
  echo "isolated build OK: $LIB contains MotorCortexController"
  echo "Production package and library untouched.  To install: bash install_robot.sh install"
  exit 0
fi

if [ "$MODE" != install ]; then echo "usage: install_robot.sh [isolated|install]" >&2; exit 2; fi
if pgrep -f '(^|/)franka_control_node([[:space:]]|$)' >/dev/null; then
  echo "Stop the Franka controller launch before installing." >&2
  exit 1
fi
BACKUP=$(mktemp -d "$HOME/motor_cortex_backup.XXXXXX")
cp -a "$PKG" "$BACKUP/panda"
cp -a "$HOME/catkin_ws/devel/lib/libpanda_controllers.so" "$BACKUP/" 2>/dev/null || true
echo "Backup: $BACKUP"
trap 'echo "Installation incomplete; restore from $BACKUP" >&2' ERR
add_to "$PKG"
build "$HOME/catkin_ws"
echo "Installed.  Nothing was started.  Follow ROBOT_COMMANDS.md."
