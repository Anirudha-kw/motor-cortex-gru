// Motor-cortex torque controller for the Franka Panda: runs the in-context GRU policy of forcefield_gru (retrained for
// the real arm by s2r.py) inside the 1 kHz real-time loop, with the reach sequencer and the exact 10 ms sensing delay
// of training (motor_cortex_core.h), a virtual force field at the handle, and safety trips.
//
// Torque sent to the robot (on top of Franka's own gravity compensation; Coriolis is NOT added -- the policy was
// trained on a plant whose Coriolis forces it had to handle itself):
//     tau = (clip(policy torque - D dq, +-tau_cap)  +  J_handle' F_field) * ramp          (running)
// D: joint damping set by "damping <k>" (default 0).  The real arm delivers torque later and weaker than the policy's
// training plants (fitted from the logged reaches, tests/plant_fit.py): without damping the shoulder and elbow
// oscillate vertically and trip; k = 12 N m s/rad on joints 1-4 (x0.2 on 5-6, x0.1 on 7) stabilises every fitted
// plant in simulation and leaves the nominal ones as they were (s2r_eval.py --ctrl-damping).
// then rate-limited to delta_tau_max per 1 ms against tau_J_d, as in training (1000 N m/s).
//
// Modes: HOLD (joint impedance around q_hold), HOMING (8 s minimum-jerk joint move to the trained start posture),
// RUNNING (an episode of centre-out-and-back reaches), TRIPPED (joint damping only, until "reset").
//
// Topics (relative to the controller's node handle):
//   command   std_msgs/String: "home" | "start [n_out] [seed]" | "stop" | "reset"
//   field     std_msgs/Float64MultiArray, 7 values: B (2x2 row-major) [N s/m] then c (3) [N]:
//             F = [B v_xy + c_xy, c_z] at the handle, |F| clipped to field_force_max
//   status    std_msgs/Int32, published on events: see MotorCortexStatus
//   telemetry std_msgs/Float64MultiArray at 100 Hz: see kTelemetry in the .cpp
//   policy_log  std_msgs/Float64MultiArray on every policy tick: time, reach, step, go, the 28 observations, u (7),
//               capped policy torque (7)
//   fast_log    std_msgs/Float64MultiArray at 1 kHz while running, during a pulse and for 1 s after: time, mode,
//               q (7), dq (7), measured tau_J (7), tau_J_d (7), torque sent (7), gravity (7), tau_ext_hat_filtered (7)
//   command "pulse <joint 1-7> <N m, |x| <= 5> <ms, 10-200>": in HOLD only, add that torque on one joint for that
//               long on top of the hold (system identification); logged on fast_log
//   command "damping <k, 0-30>": joint damping k N m s/rad (joints 1-4; x0.2 on 5-6, x0.1 on 7) added to the policy
//               torque from the next start on; status 15 when accepted
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <panda/motor_cortex_core.h>

namespace panda_controllers {

enum MotorCortexStatus : int {
  kStatusHold = 0, kStatusHoming = 1, kStatusRunning = 2, kStatusTripped = 3,
  kStatusEpisodeDone = 10, kStatusRefusedNotHome = 11, kStatusRefusedBusy = 12, kStatusRefusedTooFar = 13,
  kStatusHomeWarning = 14,   // the robot's handle at q_home differs from the training model's by > 3 mm
  kStatusDampingSet = 15,    // "damping k" accepted (in HOLD); applies from the next start
  kTripWorkspace = 21, kTripJointSpeed = 22, kTripJointLimit = 23, kTripNonFinite = 24, kTripHandleSpeed = 25,
};

class MotorCortexController : public controller_interface::MultiInterfaceController<
                                  franka_hw::FrankaModelInterface,
                                  hardware_interface::EffortJointInterface,
                                  franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  enum class Mode { kHold, kHoming, kRunning, kTripped };
  enum Command : int { kNone = 0, kHome = 1, kStart = 2, kStop = 3, kReset = 4, kPulse = 5, kDamping = 6 };
  struct Pulse {
    int joint = 0;
    double amp = 0.0, duration = 0.0;
  };
  struct EpisodePlan {             // written by the command callback (non-real-time), read when "start" is consumed
    int n_out = 6;
    std::array<double, MotorCortexEpisode::kMaxReaches> angle{};
  };
  struct Field {
    std::array<double, 7> v{};     // B11 B12 B21 B22 cx cy cz
  };

  Vector7d saturateTorqueRate(const Vector7d& tau_d, const Vector7d& tau_J_d) const;
  void enterHold(const Vector7d& q);
  void trip(int reason);
  void publishStatus(int code);
  void commandCallback(const std_msgs::StringConstPtr& msg);
  void fieldCallback(const std_msgs::Float64MultiArrayConstPtr& msg);

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  ros::Subscriber sub_command_, sub_field_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Int32>> pub_status_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>> pub_telemetry_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>> pub_policy_, pub_fast_;

  MotorCortexCore core_;
  std::atomic<int> pending_command_{kNone};
  realtime_tools::RealtimeBuffer<EpisodePlan> plan_buffer_;
  realtime_tools::RealtimeBuffer<Field> field_buffer_;
  realtime_tools::RealtimeBuffer<Pulse> pulse_buffer_;
  std::atomic<double> damping_request_{0.0};   // from the command callback
  double damping_k_{0.0};                       // accepted in HOLD by the real-time thread
  Vector7d policy_damping_ = Vector7d::Zero();  // [N m s/rad], fixed at each start
  Pulse pulse_;                         // the active pulse (HOLD only)
  double pulse_until_{-1.0}, fast_until_{-1.0}, clock_{0.0};

  // parameters
  double handle_offset_{0.10};          // handle length past the flange along its z axis [m]
  double reach_radius_{0.06};          // 0.12 (the simulation probes) drove joint speeds to Franka's limits
  double delta_tau_max_{1.0};           // per 1 ms tick [N m]
  double ramp_time_{0.1};               // policy torque ramps in over this long at an episode's start [s]
  double home_time_{8.0};
  double home_max_distance_{1.0};       // refuse homing from farther than this in any joint [rad]
  double start_tolerance_{0.05};        // |q - q_home| at "start" [rad]
  double field_force_max_{30.0};
  double handle_speed_max_{1.0}, joint_margin_{0.05};
  Eigen::Vector3d box_{0.25, 0.25, 0.15};
  Vector7d tau_cap_, k_hold_, d_hold_, d_trip_, dq_max_;   // dq_max: per-joint speed trip [rad/s]

  // state
  Mode mode_{Mode::kHold};
  Vector7d q_hold_, q_home_start_, policy_tau_;
  Eigen::Vector3d home_ = Eigen::Vector3d::Zero();
  double mode_time_{0.0};
  long tick_{0};
  long policy_ticks_{0};
  int last_event_{kStatusHold};
};

}  // namespace panda_controllers
