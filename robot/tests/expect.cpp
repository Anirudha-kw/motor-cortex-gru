// Closed-loop harness: the real controller source against a simulated Panda (MuJoCo, real-Panda joint friction,
// gravity compensated outside the controller as Franka does, 1 kHz).  Script: a start refused away from home,
// homing, an episode in a curl field (the policy's reaches are measured), a forced workspace trip, and reset.
// Checks the rate limit on every tick.  Build: see run_harness.sh.
// Every header the controller uses is included first, so that the access override below reaches only its classes.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka/robot_state.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <pluginlib/class_list_macros.h>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <panda/motor_cortex_core.h>
#define private public                       // test access to callbacks and parameters
#include "../src/motor_cortex_controller.cpp"
#undef private

#include <cstdio>
#include <cstdlib>
#include <memory>

using panda_controllers::MotorCortexController;

namespace {
std_msgs::StringConstPtr cmd(const std::string& s) { auto m = std::make_shared<std_msgs::String>(); m->data = s; return m; }
std_msgs::Float64MultiArrayConstPtr field(double curl) {
  auto m = std::make_shared<std_msgs::Float64MultiArray>();
  m->data = {0.0, curl, -curl, 0.0, 0.0, 0.0, 0.0};
  return m;
}
int last_status() { return harness::status.empty() ? -1 : harness::status.back(); }
}  // namespace


// What to expect on the robot: three episodes through the real controller (6 cm, first-session caps):
// A null field, B curl field, C curl field switched off after the 3rd outward reach (within the episode).
int main(int argc, char** argv) {
  harness::m = mj_loadModel(argv[1], nullptr);
  harness::d = mj_makeData(harness::m);
  harness::d_fk = mj_makeData(harness::m);
  mjData* d_grav = mj_makeData(harness::m);
  harness::link7 = mj_name2id(harness::m, mjOBJ_BODY, "link7");
  const double q_home[7] = {0.0, -0.35, 0.0, -2.3, 0.0, 1.95, 0.785};
  for (int i = 0; i < 7; ++i) harness::d->qpos[i] = q_home[i] + 0.05 * (i % 2 ? -1 : 1);
  mj_forward(harness::m, harness::d);
  harness::str_params["arm_id"] = "panda";
  harness::str_params["weights_path"] = argv[2];
  harness::vstr_params["joint_names"] = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                                         "panda_joint5", "panda_joint6", "panda_joint7"};
  harness::vec_params["tau_cap"] = {20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0};
  hardware_interface::RobotHW hw; ros::NodeHandle nh; MotorCortexController c;
  if (!c.init(&hw, nh)) return 1;
  c.starting(ros::Time());
  // Optional plant (comma lists of 7, e.g. from plant_fit2_top.npy): applied = gain * firstorder(command, lag) -
  // friction * tanh(dq / 0.05); DAMPING_K: "damping <k>" before every episode.
  auto list7 = [](const char* name, double dflt, double* out) {
    for (int i = 0; i < 7; ++i) out[i] = dflt;
    if (const char* v = std::getenv(name)) {
      std::stringstream ss(v); std::string item; int i = 0;
      while (std::getline(ss, item, ',') && i < 7) out[i++] = std::atof(item.c_str());
    }
  };
  double gain[7], lag[7], fric[7], alpha[7], delivered[7] = {0, 0, 0, 0, 0, 0, 0};
  list7("PLANT_GAIN", 1.0, gain); list7("PLANT_LAG", 0.0, lag); list7("PLANT_FRIC", 0.0, fric);
  for (int i = 0; i < 7; ++i) alpha[i] = lag[i] > 0 ? 1.0 - std::exp(-0.001 / lag[i]) : 1.0;
  // PLANT_DELAY_MS: a pure transport delay in front of the first-order lag (the jitter test; default 0)
  const int delay = std::getenv("PLANT_DELAY_MS") ? std::atoi(std::getenv("PLANT_DELAY_MS")) : 0;
  std::vector<std::array<double, 7>> pipe(delay + 1, std::array<double, 7>{});
  size_t pipe_k = 0;
  const char* damping = std::getenv("DAMPING_K");
  // SCHEDULE (Gate 0): decay | within | within_rev -- damping set every tick from the reach and step (GATE0.md)
  const std::string schedule = std::getenv("SCHEDULE") ? std::getenv("SCHEDULE") : "";
  auto scheduled_k = [&schedule](int reach, int step) {
    const bool moving = step >= 11 && step < 41;
    if (schedule == "decay") return reach < 2 ? 20.0 : reach < 4 ? 12.0 : 6.0;
    if (schedule == "within") return moving ? 4.0 : 20.0;
    if (schedule == "within_rev") return moving ? 20.0 : 4.0;
    return -1.0;
  };
  // PROBE_REACH=r (outward reach 0..5) and PROBE_N=F: a lateral force F (N, + along the reach's left normal) at the
  // hand for 50 ms from step 16 of that reach, in the null and curl episodes; prints the lateral position (mm) for
  // 400 ms from onset as "PTRACE <episode> ..." (impedance test: run +F and -F, halve the difference)
  const int probe_reach = std::getenv("PROBE_REACH") ? 2 * std::atoi(std::getenv("PROBE_REACH")) : -1;
  const double probe_n = std::getenv("PROBE_N") ? std::atof(std::getenv("PROBE_N")) : 0.0;
  const int probe_body = mj_name2id(harness::m, mjOBJ_BODY, "link7");
  Eigen::Vector3d probe_force = Eigen::Vector3d::Zero(), probe_point = Eigen::Vector3d::Zero();
  // DUMP=<file>: the null episode's 1 kHz joint velocities and torque commands (jitter spectra)
  std::FILE* dump = std::getenv("DUMP") ? std::fopen(std::getenv("DUMP"), "w") : nullptr;
  const double curl = 13.0 * 2.72;
  struct Ep { const char* name; double curl; int off_reach; };
  // OFF_AFTER=K: episode C switches the field off after the K-th outward reach (default 3, as the script's --off-after)
  const int off_after = std::getenv("OFF_AFTER") ? std::atoi(std::getenv("OFF_AFTER")) : 3;
  const Ep eps[3] = {{"A null field", 0.0, 99}, {"B curl field", curl, 99}, {"C curl, off after K outward", curl, 2 * off_after}};
  auto step = [&]() {
    mj_forward(harness::m, harness::d);
    c.update(ros::Time(), ros::Duration());
    for (int i = 0; i < 7; ++i) harness::tau_J_d[i] = harness::command[i];
    for (int i = 0; i < 7; ++i) { d_grav->qpos[i] = harness::d->qpos[i]; d_grav->qvel[i] = 0.0; }
    mj_forward(harness::m, d_grav);
    for (int i = 0; i < 7; ++i) pipe[pipe_k % pipe.size()][i] = harness::command[i];
    const std::array<double, 7>& late = pipe[(pipe_k + 1) % pipe.size()];   // the command from `delay` ticks ago
    ++pipe_k;
    for (int i = 0; i < 7; ++i) {
      delivered[i] += alpha[i] * (late[i] - delivered[i]);
      harness::d->qfrc_applied[i] = gain[i] * delivered[i] - fric[i] * std::tanh(harness::d->qvel[i] / 0.05) +
                                    d_grav->qfrc_bias[i];
    }
    if (probe_force.squaredNorm() > 0.0) {
      const mjtNum f[3] = {probe_force.x(), probe_force.y(), probe_force.z()}, zero[3] = {0, 0, 0},
                   p[3] = {probe_point.x(), probe_point.y(), probe_point.z()};
      mj_applyFT(harness::m, harness::d, f, zero, p, probe_body, harness::d->qfrc_applied);
    }
    mj_step(harness::m, harness::d);
  };
  for (const Ep& e : eps) {
    c.fieldCallback(field(0.0));
    c.commandCallback(cmd("reset"));                     // only acts after a trip
    for (int k = 0; k < 10; ++k) step();
    c.commandCallback(cmd("home"));
    for (int k = 0; k < 9000; ++k) step();
    if (damping) {
      c.commandCallback(cmd(std::string("damping ") + damping));
      for (int k = 0; k < 10; ++k) step();
    }
    c.fieldCallback(field(e.curl));
    // same target order in all three (SEED env: which order; default 4)
    c.commandCallback(cmd(std::string("start 6 ") + (std::getenv("SEED") ? std::getenv("SEED") : "4")));
    std::vector<std::vector<double>> rows;
    bool switched = false;
    int probe_row = -1, probe_left = 0;
    bool field_on = e.curl != 0.0;                       // FIELD_OUTWARD_ONLY=1: the field is off on return reaches
    const bool outward_only = std::getenv("FIELD_OUTWARD_ONLY") && std::atoi(std::getenv("FIELD_OUTWARD_ONLY"));
    const bool probe_here = probe_reach >= 0 && e.name[0] != 'C';
    size_t seen = harness::telemetry.size();
    for (int k = 0; k < 9500; ++k) {
      if (!schedule.empty() && c.mode_ == MotorCortexController::Mode::kRunning)
        c.policy_damping_ = scheduled_k(c.core_.reach(), c.core_.step()) * panda_controllers::kDampingShape;
      step();
      if (dump && e.name[0] == 'A' && c.mode_ == MotorCortexController::Mode::kRunning) {   // 1 kHz: dq, command
        for (int i = 0; i < 7; ++i) std::fprintf(dump, "%.6g ", harness::d->qvel[i]);
        for (int i = 0; i < 7; ++i) std::fprintf(dump, "%.6g%c", harness::command[i], i == 6 ? '\n' : ' ');
      }
      if (outward_only && e.curl != 0.0 && c.mode_ == MotorCortexController::Mode::kRunning) {
        const bool on = c.core_.reach() % 2 == 0 && c.core_.reach() < e.off_reach;
        if (on != field_on) { c.fieldCallback(field(on ? e.curl : 0.0)); field_on = on; }
      }
      if (!switched && c.mode_ == MotorCortexController::Mode::kRunning && c.core_.reach() >= e.off_reach) {
        c.fieldCallback(field(0.0));
        switched = true;
      }
      if (c.mode_ == MotorCortexController::Mode::kRunning && harness::telemetry.size() != seen) {
        seen = harness::telemetry.size();
        rows.push_back(harness::telemetry.back());       // fresh 100 Hz rows only
        if (probe_left > 0 && --probe_left == 0) probe_force.setZero();
        const auto& row = rows.back();
        if (probe_here && probe_row < 0 && static_cast<int>(row[3]) == probe_reach && c.core_.step() >= 16) {
          const Eigen::Vector3d tgt(row[6], row[7], row[8]);
          const Eigen::Vector2d dir = (tgt - c.home_).head<2>().normalized();
          probe_force << -dir.y() * probe_n, dir.x() * probe_n, 0.0;
          probe_row = static_cast<int>(rows.size()) - 1;
          probe_left = 5;
        }
        if (probe_force.squaredNorm() > 0.0) probe_point << row[9], row[10], row[11];
      }
      if (c.mode_ != MotorCortexController::Mode::kRunning && k > 100) break;
    }
    std::printf("%s (status %d):\n  outward  signed lateral @peak [mm]  peak speed [m/s]  time to peak [s]  end error [mm]\n",
                e.name, last_status());
    const Eigen::Vector3d home = c.home_;
    double max_z = 0.0, dq_max = 0.0, effort = 0.0;
    for (const auto& row : rows) {
      max_z = std::max(max_z, std::abs(row[11] - home.z()));
      for (int i = 0; i < 7; ++i) dq_max = std::max(dq_max, std::abs(row[22 + i]));
      for (int i = 0; i < 4; ++i) effort += std::abs(row[29 + i]) / 4.0;
    }
    std::printf("SUMMARY %c status %d rows %zu maxz_mm %.2f dqmax %.3f effort %.3f\n", e.name[0], last_status(),
                rows.size(), 1e3 * max_z, dq_max, rows.empty() ? 0.0 : effort / rows.size());
    probe_force.setZero();
    if (probe_row >= 0) {
      const auto& r0 = rows[probe_row];
      const Eigen::Vector3d tgt(r0[6], r0[7], r0[8]);
      const Eigen::Vector2d dir = (tgt - home).head<2>().normalized(), nrm(-dir.y(), dir.x());
      std::printf("PTRACE %c", e.name[0]);
      for (int i = probe_row; i < probe_row + 40 && i < static_cast<int>(rows.size()); ++i)
        std::printf(" %.4f", 1e3 * (Eigen::Vector2d(rows[i][9], rows[i][10]) - home.head<2>()).dot(nrm));
      std::printf("\n");
    }
    for (int r = 0; r < 12; r += 2) {
      double peak = 0, lat = 0, t_peak = 0; Eigen::Vector3d tgt, last; int first = -1, n = 0;
      for (size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        if (static_cast<int>(row[3]) != r) continue;
        if (first < 0) first = static_cast<int>(i);
        tgt << row[6], row[7], row[8];
        const Eigen::Vector3d p(row[9], row[10], row[11]), v(row[12], row[13], row[14]);
        const Eigen::Vector2d dir = (tgt - home).head<2>().normalized(), nrm(-dir.y(), dir.x());
        if (v.head<2>().norm() > peak) { peak = v.head<2>().norm(); lat = (p - home).head<2>().dot(nrm); t_peak = 0.01 * (i - first); }
        (void)n;
        last = p; ++n;
      }
      static double null_lat[6];
      if (e.curl == 0.0) null_lat[r / 2] = lat;
      std::printf("  %d%s       %+6.2f   (minus null %+5.2f)      %.2f              %.2f             %.2f\n", r / 2,
                  (e.off_reach < 99 && r >= e.off_reach) ? " (off)" : "      ", 1e3 * lat, 1e3 * (lat - null_lat[r / 2]),
                  peak, t_peak, 1e3 * (last - tgt).norm());
    }
  }
  return 0;
}
