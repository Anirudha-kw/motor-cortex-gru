#pragma once
#include <string>
#include <franka/robot_state.h>
#include "harness_world.h"
namespace franka_hw {
struct FrankaStateHandle {
  franka::RobotState getRobotState() const {
    franka::RobotState s;
    for (int i = 0; i < 7; ++i) { s.q[i] = harness::d->qpos[i]; s.dq[i] = harness::d->qvel[i]; s.tau_J_d[i] = harness::tau_J_d[i]; s.tau_J[i] = harness::tau_J[i]; }
    s.O_T_EE = harness::flange_pose(harness::d);
    for (int i = 0; i < 16; ++i) { s.F_T_EE[i] = (i % 5 == 0); s.EE_T_K[i] = (i % 5 == 0); }
    return s;
  }
};
struct FrankaStateInterface { FrankaStateHandle getHandle(const std::string&) { return FrankaStateHandle(); } };
}
