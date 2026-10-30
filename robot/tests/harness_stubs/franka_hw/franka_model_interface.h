#pragma once
#include <array>
#include <string>
#include <franka/robot_state.h>
#include "harness_world.h"
namespace franka_hw {
struct FrankaModelHandle {
  std::array<double, 16> getPose(const franka::Frame&) const { return harness::flange_pose(harness::d); }
  std::array<double, 16> getPose(const franka::Frame&, const std::array<double, 7>& q, const std::array<double, 16>&,
                                 const std::array<double, 16>&) const {
    for (int i = 0; i < 7; ++i) harness::d_fk->qpos[i] = q[i];
    mj_kinematics(harness::m, harness::d_fk);
    return harness::flange_pose(harness::d_fk);
  }
  std::array<double, 42> getZeroJacobian(const franka::Frame&) const { return harness::flange_jacobian(harness::d); }
  std::array<double, 7> getGravity() const { return harness::gravity; }
};
struct FrankaModelInterface { FrankaModelHandle getHandle(const std::string&) { return FrankaModelHandle(); } };
}
