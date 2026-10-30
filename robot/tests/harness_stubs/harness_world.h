#pragma once
// Globals of the closed-loop harness: the simulated Panda the stubs below read from and write to.
#include <array>
#include <map>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
namespace harness {
inline mjModel* m = nullptr;
inline mjData* d = nullptr;       // the plant
inline mjData* d_fk = nullptr;    // kinematics at arbitrary joint angles
inline int link7 = -1;
constexpr double kFlange = 0.107; // link 7 origin -> flange, along its z axis
inline std::array<double, 7> command{}, tau_J_d{};
inline std::map<std::string, std::string> str_params;
inline std::map<std::string, std::vector<double>> vec_params;
inline std::map<std::string, double> num_params;
inline std::map<std::string, std::vector<std::string>> vstr_params;
inline std::vector<int> status;
inline std::vector<std::vector<double>> telemetry, policy_log, fast_log;
inline std::array<double, 7> tau_J{}, gravity{};   // measured joint torque and Franka's gravity model
inline std::array<double, 16> flange_pose(const mjData* dd) {      // column-major, like libfranka
  const double* R = dd->xmat + 9 * link7;
  const double* p = dd->xpos + 3 * link7;
  std::array<double, 16> T{};
  for (int c = 0; c < 3; ++c)
    for (int r = 0; r < 3; ++r) T[4 * c + r] = R[3 * r + c];
  for (int r = 0; r < 3; ++r) T[12 + r] = p[r] + R[3 * r + 2] * kFlange;
  T[15] = 1.0;
  return T;
}
inline std::array<double, 42> flange_jacobian(const mjData* dd) {  // column-major 6x7 zero Jacobian
  const std::array<double, 16> T = flange_pose(dd);
  const double point[3] = {T[12], T[13], T[14]};
  std::vector<double> jp(3 * m->nv), jr(3 * m->nv);
  mj_jac(m, dd, jp.data(), jr.data(), point, link7);
  std::array<double, 42> J{};
  for (int c = 0; c < 7; ++c)
    for (int r = 0; r < 3; ++r) {
      J[6 * c + r] = jp[r * m->nv + c];
      J[6 * c + 3 + r] = jr[r * m->nv + c];
    }
  return J;
}
}  // namespace harness
