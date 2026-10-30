#pragma once
#include <string>
#include <franka/robot_state.h>
namespace franka_hw {
struct FrankaStateHandle { franka::RobotState s; const franka::RobotState& getRobotState() const { return s; } };
struct FrankaStateInterface { FrankaStateHandle getHandle(const std::string&) { return FrankaStateHandle(); } };
}
