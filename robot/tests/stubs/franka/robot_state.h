#pragma once
#include <array>
namespace franka {
enum class Frame { kJoint1, kFlange, kEndEffector, kStiffness };
struct RobotState { std::array<double, 7> q{}, dq{}, tau_J_d{}, tau_J{}, tau_ext_hat_filtered{}; std::array<double, 16> O_T_EE{}, F_T_EE{}, EE_T_K{}; };
}
