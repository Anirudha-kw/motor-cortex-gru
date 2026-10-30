#pragma once
#include <array>
#include <string>
#include <franka/robot_state.h>
namespace franka_hw {
struct FrankaModelHandle {
  std::array<double, 16> getPose(const franka::Frame&) const { return {}; }
  std::array<double, 16> getPose(const franka::Frame&, const std::array<double, 7>&, const std::array<double, 16>&,
                                 const std::array<double, 16>&) const { return {}; }
  std::array<double, 42> getZeroJacobian(const franka::Frame&) const { return {}; }
  std::array<double, 7> getGravity() const { return {}; }
};
struct FrankaModelInterface { FrankaModelHandle getHandle(const std::string&) { return FrankaModelHandle(); } };
}
