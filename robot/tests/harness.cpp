// Closed-loop harness: the real controller source against a simulated Panda (MuJoCo, real-Panda joint friction,
// gravity compensated outside the controller as Franka does, 1 kHz).  Script: a start refused away from home,
// homing, an episode in a curl field (the policy's reaches are measured), a forced workspace trip, reset, and a
// system-identification pulse in hold (with the policy and 1 kHz logs checked).
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

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: harness model.mjb weights.mcgru\n"); return 2; }
  char err[1000];
  harness::m = mj_loadModel(argv[1], nullptr);
  if (!harness::m) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }
  harness::d = mj_makeData(harness::m);
  harness::d_fk = mj_makeData(harness::m);
  mjData* d_grav = mj_makeData(harness::m);
  harness::link7 = mj_name2id(harness::m, mjOBJ_BODY, "link7");
  (void)err;
  const double q_home[7] = {0.0, -0.35, 0.0, -2.3, 0.0, 1.95, 0.785};
  const double offset[7] = {0.3, -0.2, 0.2, 0.2, 0.1, -0.1, 0.2};
  for (int i = 0; i < 7; ++i) harness::d->qpos[i] = q_home[i] + offset[i];
  mj_forward(harness::m, harness::d);

  harness::str_params["arm_id"] = "panda";
  harness::str_params["weights_path"] = argv[2];
  harness::vstr_params["joint_names"] = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                                         "panda_joint5", "panda_joint6", "panda_joint7"};
  if (std::getenv("FIRST_SESSION")) harness::vec_params["tau_cap"] = {20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0};
  // REAL_FIT=0-3: torque delivery of a plant fitted to the logged real reach (s2r_eval.REAL_FITS; build its model with
  // ARMATURE=<fit's> TOOL=0.63 FRICTION_SCALE=0 DAMPING=0 python3 make_harness_model.py out.mjb): per joint
  // applied = gain * firstorder(command, lag) - friction * tanh(dq / 0.05).  Default: ideal delivery.
  const double kFits[4][3][7] = {
      {{0.83, 0.62, 0.99, 1.02, 0.84, 0.96, 0.54}, {.018, .023, .037, .031, .015, .003, .029},
       {0.47, 1.31, 0.12, 0.58, 0.3, 0.81, 0.01}},
      {{0.66, 0.82, 0.86, 1.02, 0.89, 0.73, 0.55}, {.003, .033, .039, .036, .004, .01, .008},
       {1.2, 0.75, 1.94, 0.87, 0.52, 0.29, 0.01}},
      {{0.83, 0.65, 0.99, 0.98, 1.04, 0.58, 0.94}, {.015, .002, .01, .032, .004, .004, .019},
       {0.71, 1.95, 1.4, 0.12, 0.05, 0.86, 0.37}},
      {{1.04, 0.86, 0.51, 0.77, 0.97, 0.91, 0.55}, {.025, .026, .005, .029, .007, .003, .011},
       {0.98, 1.05, 0.79, 0.27, 0.15, 0.92, 0.47}}};
  double gain[7], alpha[7], fric[7], delivered[7] = {0, 0, 0, 0, 0, 0, 0};
  const int fit = std::getenv("REAL_FIT") ? std::atoi(std::getenv("REAL_FIT")) : -1;
  for (int i = 0; i < 7; ++i) {
    gain[i] = fit >= 0 ? kFits[fit][0][i] : 1.0;
    alpha[i] = fit >= 0 ? 1.0 - std::exp(-0.001 / kFits[fit][1][i]) : 1.0;
    fric[i] = fit >= 0 ? kFits[fit][2][i] : 0.0;
  }
  const char* damping = std::getenv("DAMPING_K");      // sent as "damping <k>" before the curl episode
  if (fit >= 0) std::printf("  plant: real fit %d; controller damping %s\n", fit, damping ? damping : "0");
  hardware_interface::RobotHW hw;
  ros::NodeHandle nh;
  MotorCortexController c;
  if (!c.init(&hw, nh)) { std::fprintf(stderr, "init failed\n"); return 1; }
  c.starting(ros::Time());
  if (const char* r = std::getenv("REACH_RADIUS")) c.reach_radius_ = std::atof(r);
  std::printf("  reach radius %.3f m\n", c.reach_radius_);

  const double dt = harness::m->opt.timestep;
  int failures = 0, rate_violations = 0;
  auto expect = [&failures](bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    failures += !ok;
  };
  std::vector<std::vector<double>> episode;           // telemetry rows while the curl episode runs
  bool homed = false, ran = false, tripped = false;
  double t_done = -1.0, t_trip = -1.0;
  size_t policy_rows_episode = 0, fast_rows_before_pulse = 0;
  double pulse_peak = 0.0, q4_hold = 0.0;
  for (long k = 0; k < 31000; ++k) {
    const double t = k * dt;
    if (k == 200) c.commandCallback(cmd("start 3 7"));
    if (k == 300) { expect(last_status() == panda_controllers::kStatusRefusedNotHome, "start refused away from home"); c.commandCallback(cmd("home")); }
    if (k == 9300 && damping) c.commandCallback(cmd(std::string("damping ") + damping));
    if (k == 9400 && damping) expect(last_status() == panda_controllers::kStatusDampingSet, "damping accepted in hold");
    if (k == 9500) {
      double worst = 0.0;
      for (int i = 0; i < 7; ++i) worst = std::max(worst, std::abs(harness::d->qpos[i] - q_home[i]));
      homed = worst < 0.01 && c.mode_ == MotorCortexController::Mode::kHold;
      std::printf("  homing: max |q - q_home| %.4f rad\n", worst);
      expect(homed, "homed and holding");
      c.fieldCallback(field(13.0 * 2.72));
      c.commandCallback(cmd("start 3 7"));
    }
    if (k > 9500 && k < 16000 && c.mode_ == MotorCortexController::Mode::kRunning && !harness::telemetry.empty())
      episode.push_back(harness::telemetry.back());
    if (k > 9600 && !ran && last_status() == panda_controllers::kStatusEpisodeDone) { ran = true; t_done = t; }
    if (k == 16000) {
      expect(ran, "episode finished (status 10)");
      policy_rows_episode = harness::policy_log.size();
      c.fieldCallback(field(0.0));
      c.commandCallback(cmd("home"));                  // the null space drifts during an episode: home again
    }
    if (k == 25000) {
      c.box_ = Eigen::Vector3d(0.004, 0.004, 0.004);    // force a workspace trip during the first reach
      c.commandCallback(cmd("start 2 1"));
    }
    if (k > 25000 && !tripped && last_status() == panda_controllers::kTripWorkspace) { tripped = true; t_trip = t; }
    if (k == 28000) {
      expect(tripped && c.mode_ == MotorCortexController::Mode::kTripped, "workspace trip -> damping only");
      double v = 0.0;
      for (int i = 0; i < 7; ++i) v = std::max(v, std::abs(harness::d->qvel[i]));
      std::printf("  max |dq| 3 s after the trip %.4f rad/s (trip at %.2f s)\n", v, t_trip);
      c.commandCallback(cmd("reset"));
    }
    if (k == 28500) expect(c.mode_ == MotorCortexController::Mode::kHold && last_status() == 0, "reset -> hold");
    if (k == 28900) c.commandCallback(cmd("pulse 4 9 100"));          // over the 5 N m limit: ignored
    if (k == 28950) expect(c.pulse_until_ < c.clock_, "out-of-range pulse ignored");
    if (k == 29000) {
      fast_rows_before_pulse = harness::fast_log.size();
      q4_hold = harness::d->qpos[3];
      c.commandCallback(cmd("pulse 4 3 100"));
    }
    if (k > 29000) pulse_peak = std::max(pulse_peak, std::abs(harness::d->qpos[3] - q4_hold));
    if (k == 30500) {
      const size_t n = harness::fast_log.size() - fast_rows_before_pulse;
      std::printf("  pulse 3 N m x 100 ms on joint 4: peak deflection %.2f mrad, %zu fast-log rows\n", 1e3 * pulse_peak, n);
      expect(pulse_peak > 1e-4 && pulse_peak < 0.05, "pulse deflects joint 4 a little under the stiff hold");
      expect(n >= 1090 && n <= 1110, "fast log covers the pulse and 1 s after it");
      expect(std::abs(harness::d->qpos[3] - q4_hold) < 2e-3, "joint 4 back at hold after the pulse");
    }

    // the robot: state -> controller -> torque (+ gravity compensation) -> physics
    mj_forward(harness::m, harness::d);
    for (int i = 0; i < 7; ++i) { d_grav->qpos[i] = harness::d->qpos[i]; d_grav->qvel[i] = 0.0; }
    mj_forward(harness::m, d_grav);
    for (int i = 0; i < 7; ++i) {
      harness::gravity[i] = d_grav->qfrc_bias[i];
      harness::tau_J[i] = harness::tau_J_d[i] + harness::gravity[i];   // ideal torque sensor
    }
    const std::array<double, 7> previous = harness::tau_J_d;
    c.update(ros::Time(), ros::Duration());
    for (int i = 0; i < 7; ++i) {
      if (std::abs(harness::command[i] - previous[i]) > c.delta_tau_max_ + 1e-9) ++rate_violations;
      harness::tau_J_d[i] = harness::command[i];
    }
    for (int i = 0; i < 7; ++i) { d_grav->qpos[i] = harness::d->qpos[i]; d_grav->qvel[i] = 0.0; }
    mj_forward(harness::m, d_grav);                    // gravity torque = qfrc_bias at zero velocity
    for (int i = 0; i < 7; ++i) {
      delivered[i] += alpha[i] * (harness::command[i] - delivered[i]);
      harness::d->qfrc_applied[i] = gain[i] * delivered[i] - fric[i] * std::tanh(harness::d->qvel[i] / 0.05) +
                                    d_grav->qfrc_bias[i];
    }
    mj_step(harness::m, harness::d);
  }
  expect(rate_violations == 0, "torque rate limited to delta_tau_max every tick");
  // policy log: one row per policy tick of the curl episode (3 out-and-back = 6 reaches x 70 ticks)
  std::printf("  policy log: %zu rows in the curl episode, %zu in all; fast log: %zu rows\n", policy_rows_episode,
              harness::policy_log.size(), harness::fast_log.size());
  expect(policy_rows_episode == 6 * 70, "policy log has one row per policy tick");
  bool obs_ok = !harness::policy_log.empty();
  for (const auto& row : harness::policy_log) obs_ok = obs_ok && row.size() == 46 && row[7] == row[3];   // go
  expect(obs_ok, "policy log rows carry the observed go cue");
  bool fast_ok = !harness::fast_log.empty();
  for (const auto& row : harness::fast_log) fast_ok = fast_ok && row.size() == 51;
  expect(fast_ok, "fast log rows have 51 values");
  std::printf("  status history:");
  for (int st : harness::status) std::printf(" %d", st);
  std::printf("\n");
  double dq_peak = 0.0, v_peak = 0.0;
  Eigen::Matrix<double, 7, 1> dq_joint = Eigen::Matrix<double, 7, 1>::Zero();
  for (const auto& row : episode) {
    for (int i = 0; i < 7; ++i) dq_joint(i) = std::max(dq_joint(i), std::abs(row[22 + i]));
    v_peak = std::max(v_peak, Eigen::Vector3d(row[12], row[13], row[14]).norm());
  }
  dq_peak = dq_joint.maxCoeff();
  Eigen::Matrix<double, 7, 1> tau_peak = Eigen::Matrix<double, 7, 1>::Zero();
  for (const auto& row : episode)
    for (int i = 0; i < 7; ++i) tau_peak(i) = std::max(tau_peak(i), std::abs(row[29 + i]));
  std::printf("  episode peak |commanded torque| per joint: %s N m\n",
              [&] { std::ostringstream o; o.precision(3); o << tau_peak.transpose(); return o.str(); }().c_str());
  std::printf("  episode peaks: |dq| per joint %s rad/s (max %.2f), handle speed %.2f m/s\n",
              [&] { std::ostringstream o; o << dq_joint.transpose(); return o.str(); }().c_str(), dq_peak, v_peak);

  // The curl episode: lateral error at peak speed of each outward reach (telemetry layout: see the .cpp).
  std::printf("  episode: %zu telemetry rows, done at %.2f s\n", episode.size(), t_done);
  const Eigen::Vector3d home = c.home_;
  double worst_final = 0.0;
  for (int r = 0; r < 6; r += 2) {
    double peak = 0.0, lateral = 0.0;
    Eigen::Vector3d tgt, last;
    for (const auto& row : episode) {
      if (static_cast<int>(row[3]) != r) continue;
      tgt << row[6], row[7], row[8];
      const Eigen::Vector3d p(row[9], row[10], row[11]), v(row[12], row[13], row[14]);
      Eigen::Vector2d dir = (tgt - home).head<2>().normalized(), n(-dir.y(), dir.x());
      if (v.head<2>().norm() > peak) { peak = v.head<2>().norm(); lateral = std::abs((p - home).head<2>().dot(n)); }
      last = p;
    }
    worst_final = std::max(worst_final, (last - tgt).norm());
    std::printf("  outward reach %d: peak speed %.2f m/s, lateral error at peak %.1f mm, end error %.1f mm\n",
                r / 2, peak, 1e3 * lateral, 1e3 * (last - tgt).norm());
  }
  expect(worst_final < 0.01, "every outward reach ends within 10 mm of its target");
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
