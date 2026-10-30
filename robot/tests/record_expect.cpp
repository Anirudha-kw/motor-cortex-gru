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


// Records what the robot should do, for render_expect.py: the real controller in the simulated loop, through
// homing, a null-field episode, a curl episode, and a curl episode switched off after the 3rd outward reach.
// Writes one line per fresh 100 Hz telemetry row: sim time, segment, curl gain, then the telemetry row.
int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: record_expect model.mjb weights.mcgru out.csv [gravity_model.mjb]\n");
    return 2;
  }
  harness::m = mj_loadModel(argv[1], nullptr);
  harness::d = mj_makeData(harness::m);
  harness::d_fk = mj_makeData(harness::m);
  // Franka compensates gravity with the load it was told about: a model without a mass that is on the arm leaves
  // that mass's weight uncompensated (an undeclared load).
  mjModel* m_grav = argc == 5 ? mj_loadModel(argv[4], nullptr) : harness::m;
  mjData* d_grav = mj_makeData(m_grav);
  harness::link7 = mj_name2id(harness::m, mjOBJ_BODY, "link7");
  const double q_home[7] = {0.0, -0.35, 0.0, -2.3, 0.0, 1.95, 0.785};
  const double offset[7] = {0.3, -0.2, 0.2, 0.2, 0.1, -0.1, 0.2};
  for (int i = 0; i < 7; ++i) harness::d->qpos[i] = q_home[i] + offset[i];
  mj_forward(harness::m, harness::d);
  harness::str_params["arm_id"] = "panda";
  harness::str_params["weights_path"] = argv[2];
  harness::vstr_params["joint_names"] = {"panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
                                         "panda_joint5", "panda_joint6", "panda_joint7"};
  harness::vec_params["tau_cap"] = {20.0, 20.0, 20.0, 20.0, 5.0, 5.0, 5.0};     // first-session caps
  if (const char* caps = std::getenv("CAPS")) {                                   // e.g. "40,40,40,40,8,8,8"
    std::vector<double> v; std::stringstream ss(caps); std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::atof(item.c_str()));
    harness::vec_params["tau_cap"] = v;
  }
  hardware_interface::RobotHW hw; ros::NodeHandle nh; MotorCortexController c;
  if (!c.init(&hw, nh)) return 1;
  c.starting(ros::Time());
  std::FILE* out = std::fopen(argv[3], "w");
  const double dt = harness::m->opt.timestep, curl = 13.0 * 2.72;
  double t = 0.0, field_now = 0.0;
  const double torque_gain = std::getenv("TORQUE_GAIN") ? std::atof(std::getenv("TORQUE_GAIN")) : 1.0;
  size_t seen = harness::telemetry.size();
  int seg = 0;
  auto step = [&]() {
    mj_forward(harness::m, harness::d);
    c.update(ros::Time(), ros::Duration());
    for (int i = 0; i < 7; ++i) harness::tau_J_d[i] = harness::command[i];
    for (int i = 0; i < 7; ++i) { d_grav->qpos[i] = harness::d->qpos[i]; d_grav->qvel[i] = 0.0; }
    mj_forward(m_grav, d_grav);
    for (int i = 0; i < 7; ++i) harness::d->qfrc_applied[i] = torque_gain * harness::command[i] + d_grav->qfrc_bias[i];
    mj_step(harness::m, harness::d);
    t += dt;
    if (harness::telemetry.size() != seen) {
      seen = harness::telemetry.size();
      std::fprintf(out, "%.3f %d %.3f", t, seg, field_now);
      for (double v : harness::telemetry.back()) std::fprintf(out, " %.6g", v);
      std::fprintf(out, "\n");
    }
  };
  auto set_field = [&](double g) { field_now = g; c.fieldCallback(field(g)); };
  auto home = [&]() {
    set_field(0.0);
    c.commandCallback(cmd("home"));
    for (int k = 0; k < 9000; ++k) step();
  };
  auto episode = [&](double g, int off_reach) {
    set_field(g);
    c.commandCallback(cmd("start 6 4"));
    step();
    while (c.mode_ == MotorCortexController::Mode::kRunning) {
      if (field_now != 0.0 && c.core_.reach() >= off_reach) set_field(0.0);
      step();
    }
    for (int k = 0; k < 300; ++k) step();             // a moment of hold after the episode
  };
  for (int k = 0; k < 500; ++k) step();               // hold where it starts
  seg = 1; home();
  seg = 2; episode(0.0, 99);
  seg = 3; home();
  seg = 4; episode(curl, 99);
  seg = 5; home();
  seg = 6; episode(curl, 6);
  std::fclose(out);
  std::printf("recorded %.1f s; last status %d\n", t, last_status());
  return 0;
}
