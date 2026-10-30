// Motor-cortex torque controller.  See the header for the command interface and the torque law.
#include <panda/motor_cortex_controller.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace panda_controllers {

namespace {

// Panda joint limits [rad] (Franka datasheet).
const Vector7d kQMin = (Vector7d() << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973).finished();
const Vector7d kQMax = (Vector7d() << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973).finished();
const Vector7d kTauLimit = (Vector7d() << 87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0).finished();

// Telemetry, 100 Hz: time, mode, last event, reach, step, go, target (3), handle (3), handle velocity (3), q (7),
// dq (7), commanded torque before the rate limit (7), policy output u (7), field force (3).
constexpr int kTelemetry = 6 + 3 + 3 + 3 + 7 + 7 + 7 + 7 + 3;
constexpr int kPolicyLog = 4 + 28 + 7 + 7;     // time, reach, step, go, observations, u, capped policy torque
const Vector7d kDampingShape = (Vector7d() << 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.1).finished();   // x "damping k"
constexpr int kFastLog = 2 + 7 * 7;            // time, mode, q, dq, tau_J, tau_J_d, torque sent, gravity, tau_ext

bool readVector7(ros::NodeHandle& nh, const std::string& name, Vector7d* out) {
  std::vector<double> v;
  if (!nh.getParam(name, v)) return true;          // keep the default
  if (v.size() != 7) {
    ROS_ERROR_STREAM("MotorCortexController: parameter " << name << " must have 7 values");
    return false;
  }
  for (int i = 0; i < 7; ++i) (*out)(i) = v[i];
  return true;
}

}  // namespace

bool MotorCortexController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
  std::string arm_id, weights_path;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("MotorCortexController: could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR("MotorCortexController: invalid or missing joint_names");
    return false;
  }
  if (!node_handle.getParam("weights_path", weights_path)) {
    ROS_ERROR("MotorCortexController: missing parameter weights_path (an .mcgru file from export_weights.py)");
    return false;
  }
  std::string error;
  if (!core_.load(weights_path, &error)) {
    ROS_ERROR_STREAM("MotorCortexController: cannot load the policy: " << error);
    return false;
  }

  tau_cap_ << 40.0, 40.0, 40.0, 40.0, 8.0, 8.0, 8.0;
  k_hold_ << 300.0, 300.0, 300.0, 300.0, 100.0, 80.0, 40.0;
  d_hold_ << 30.0, 30.0, 30.0, 30.0, 10.0, 8.0, 5.0;
  d_trip_ << 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 1.0;
  dq_max_ << 1.85, 1.85, 1.85, 1.85, 2.2, 2.2, 2.2;          // ~85% of the Panda's 2.175 / 2.61 rad/s limits
  if (!readVector7(node_handle, "tau_cap", &tau_cap_) || !readVector7(node_handle, "k_hold", &k_hold_) ||
      !readVector7(node_handle, "d_hold", &d_hold_) || !readVector7(node_handle, "d_trip", &d_trip_) ||
      !readVector7(node_handle, "dq_max", &dq_max_)) {
    return false;
  }
  if ((tau_cap_.array() <= 0.0).any() || (tau_cap_.array() > kTauLimit.array()).any()) {
    ROS_ERROR("MotorCortexController: tau_cap must be positive and within the Panda's joint torque limits");
    return false;
  }
  node_handle.param("handle_offset", handle_offset_, handle_offset_);
  node_handle.param("reach_radius", reach_radius_, reach_radius_);
  node_handle.param("delta_tau_max", delta_tau_max_, delta_tau_max_);
  node_handle.param("ramp_time", ramp_time_, ramp_time_);
  node_handle.param("home_time", home_time_, home_time_);
  node_handle.param("home_max_distance", home_max_distance_, home_max_distance_);
  node_handle.param("start_tolerance", start_tolerance_, start_tolerance_);
  node_handle.param("field_force_max", field_force_max_, field_force_max_);
  node_handle.param("handle_speed_max", handle_speed_max_, handle_speed_max_);
  node_handle.param("joint_margin", joint_margin_, joint_margin_);
  std::vector<double> box;
  if (node_handle.getParam("workspace_box", box) && box.size() == 3) box_ << box[0], box[1], box[2];
  if (delta_tau_max_ <= 0.0 || delta_tau_max_ > 1.0 || reach_radius_ <= 0.0 || reach_radius_ > 0.12 ||
      home_time_ < 2.0 || ramp_time_ < 0.0) {
    ROS_ERROR("MotorCortexController: delta_tau_max in (0, 1], reach_radius in (0, 0.12], home_time >= 2 s");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  auto* effort_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (model_interface == nullptr || state_interface == nullptr || effort_interface == nullptr) {
    ROS_ERROR("MotorCortexController: missing model, state or effort-joint interface");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(model_interface->getHandle(arm_id + "_model"));
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(state_interface->getHandle(arm_id + "_robot"));
    for (size_t i = 0; i < 7; ++i) joint_handles_.push_back(effort_interface->getHandle(joint_names[i]));
  } catch (const hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM("MotorCortexController: exception getting handles: " << ex.what());
    return false;
  }

  sub_command_ = node_handle.subscribe("command", 10, &MotorCortexController::commandCallback, this,
                                       ros::TransportHints().reliable().tcpNoDelay());
  sub_field_ = node_handle.subscribe("field", 10, &MotorCortexController::fieldCallback, this,
                                     ros::TransportHints().reliable().tcpNoDelay());
  pub_status_ = std::make_unique<realtime_tools::RealtimePublisher<std_msgs::Int32>>(node_handle, "status", 20, true);
  pub_telemetry_ =
      std::make_unique<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>>(node_handle, "telemetry", 50);
  pub_telemetry_->msg_.data.assign(kTelemetry, 0.0);
  pub_policy_ =
      std::make_unique<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>>(node_handle, "policy_log", 100);
  pub_policy_->msg_.data.assign(kPolicyLog, 0.0);
  pub_fast_ =
      std::make_unique<realtime_tools::RealtimePublisher<std_msgs::Float64MultiArray>>(node_handle, "fast_log", 2000);
  pub_fast_->msg_.data.assign(kFastLog, 0.0);
  pulse_buffer_.writeFromNonRT(Pulse());
  field_buffer_.writeFromNonRT(Field());
  plan_buffer_.writeFromNonRT(EpisodePlan());
  policy_tau_.setZero();
  ROS_INFO_STREAM("MotorCortexController: loaded " << weights_path << " (GRU, " << core_.hidden()
                  << " hidden units); torque caps " << tau_cap_.transpose());
  return true;
}

void MotorCortexController::starting(const ros::Time& /*time*/) {
  const franka::RobotState rs = state_handle_->getRobotState();
  std::array<double, 7> q_home{};
  for (int i = 0; i < 7; ++i) q_home[i] = core_.q_home()(i);
  const std::array<double, 16> pose = model_handle_->getPose(franka::Frame::kFlange, q_home, rs.F_T_EE, rs.EE_T_K);
  const Eigen::Affine3d O_T_F(Eigen::Matrix4d::Map(pose.data()));
  home_ = O_T_F.translation() + O_T_F.linear() * Eigen::Vector3d(0.0, 0.0, handle_offset_);
  enterHold(Eigen::Map<const Vector7d>(rs.q.data()));
  tick_ = 0;
  if ((home_ - core_.home_sim()).norm() > 0.003) publishStatus(kStatusHomeWarning);
}

void MotorCortexController::update(const ros::Time& time, const ros::Duration& period) {
  const double dt = std::min(std::max(period.toSec(), 0.0), 0.01);
  mode_time_ += dt;
  clock_ += dt;
  const franka::RobotState rs = state_handle_->getRobotState();
  const Eigen::Map<const Vector7d> q(rs.q.data()), dq(rs.dq.data()), tau_J_d(rs.tau_J_d.data());
  const std::array<double, 16> pose = model_handle_->getPose(franka::Frame::kFlange);
  const Eigen::Affine3d O_T_F(Eigen::Matrix4d::Map(pose.data()));
  const Eigen::Vector3d r = O_T_F.linear() * Eigen::Vector3d(0.0, 0.0, handle_offset_);
  const Eigen::Vector3d handle = O_T_F.translation() + r;
  const std::array<double, 42> jacobian = model_handle_->getZeroJacobian(franka::Frame::kFlange);
  const Eigen::Map<const Eigen::Matrix<double, 6, 7>> J(jacobian.data());
  Eigen::Matrix3d skew;
  skew << 0.0, -r.z(), r.y(), r.z(), 0.0, -r.x(), -r.y(), r.x(), 0.0;
  const Eigen::Matrix<double, 3, 7> J_handle = J.topRows<3>() - skew * J.bottomRows<3>();
  const Eigen::Vector3d v_handle = J_handle * dq;
  const Vector7d& q_home = core_.q_home();

  // Commands from the callback (consumed here, in the real-time thread).
  switch (pending_command_.exchange(kNone)) {
    case kReset:
      if (mode_ == Mode::kTripped) {
        enterHold(q);
        publishStatus(kStatusHold);
      }
      break;
    case kStop:
      if (mode_ != Mode::kTripped) {
        enterHold(q);
        publishStatus(kStatusHold);
      }
      break;
    case kHome:
      if (mode_ != Mode::kHold) {
        publishStatus(kStatusRefusedBusy);
      } else if ((q - q_home).cwiseAbs().maxCoeff() > home_max_distance_) {
        publishStatus(kStatusRefusedTooFar);
      } else {
        q_home_start_ = q;
        mode_ = Mode::kHoming;
        mode_time_ = 0.0;
        publishStatus(kStatusHoming);
      }
      break;
    case kPulse:
      if (mode_ == Mode::kHold) {
        pulse_ = *pulse_buffer_.readFromRT();
        pulse_until_ = clock_ + pulse_.duration;
        fast_until_ = pulse_until_ + 1.0;
      } else {
        publishStatus(kStatusRefusedBusy);
      }
      break;
    case kDamping:
      if (mode_ == Mode::kHold) {
        damping_k_ = damping_request_.load();
        publishStatus(kStatusDampingSet);
      } else {
        publishStatus(kStatusRefusedBusy);
      }
      break;
    case kStart:
      if (mode_ != Mode::kHold) {
        publishStatus(kStatusRefusedBusy);
      } else if ((q - q_home).cwiseAbs().maxCoeff() > start_tolerance_ || dq.cwiseAbs().maxCoeff() > 0.05) {
        publishStatus(kStatusRefusedNotHome);
      } else {
        const EpisodePlan& plan = *plan_buffer_.readFromRT();
        MotorCortexEpisode episode;
        episode.n_reaches = 2 * plan.n_out;
        for (int i = 0; i < plan.n_out; ++i) {
          episode.targets[2 * i] =
              home_ + reach_radius_ * Eigen::Vector3d(std::cos(plan.angle[i]), std::sin(plan.angle[i]), 0.0);
          episode.targets[2 * i + 1] = home_;
        }
        MotorCortexSense now;
        now.handle = handle;
        now.q = q;
        now.dq = dq;
        core_.begin(episode, home_, now);
        policy_damping_ = damping_k_ * kDampingShape;
        policy_tau_.setZero();
        policy_ticks_ = 0;
        mode_ = Mode::kRunning;
        mode_time_ = 0.0;
        publishStatus(kStatusRunning);
      }
      break;
    default:
      break;
  }

  // Safety trips while the controller moves the arm on its own.
  if (mode_ == Mode::kHoming || mode_ == Mode::kRunning) {
    if (((handle - home_).cwiseAbs().array() > box_.array()).any() && mode_ == Mode::kRunning) {
      trip(kTripWorkspace);
    } else if ((dq.cwiseAbs().array() > dq_max_.array()).any()) {
      trip(kTripJointSpeed);
    } else if (((q - kQMin).array() < joint_margin_).any() || ((kQMax - q).array() < joint_margin_).any()) {
      trip(kTripJointLimit);
    } else if (v_handle.norm() > handle_speed_max_) {
      trip(kTripHandleSpeed);
    }
  }

  Vector7d tau = Vector7d::Zero();
  Eigen::Vector3d force = Eigen::Vector3d::Zero();
  switch (mode_) {
    case Mode::kHold:
      tau = k_hold_.cwiseProduct(q_hold_ - q) - d_hold_.cwiseProduct(dq);
      if (clock_ < pulse_until_) tau(pulse_.joint) += pulse_.amp;      // system-identification pulse
      break;
    case Mode::kHoming: {
      const double s = std::min(1.0, mode_time_ / home_time_);
      const double p = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
      const double pd = s < 1.0 ? 30.0 * s * s * (1.0 - s) * (1.0 - s) / home_time_ : 0.0;
      const Vector7d delta = q_home - q_home_start_;
      tau = k_hold_.cwiseProduct(q_home_start_ + p * delta - q) + d_hold_.cwiseProduct(pd * delta - dq);
      if (mode_time_ > home_time_ + 0.5) {
        enterHold(q_home);
        publishStatus(kStatusHold);
      }
      break;
    }
    case Mode::kRunning: {
      if (policy_ticks_ % 10 == 0) {                 // the policy's 10 ms period
        if (!core_.running()) {
          enterHold(q);
          publishStatus(kStatusEpisodeDone);
          break;
        }
        MotorCortexSense now;
        now.handle = handle;
        now.q = q;
        now.dq = dq;
        policy_tau_ = core_.tick(now).cwiseMax(-tau_cap_).cwiseMin(tau_cap_);
        if (!core_.finite() || !policy_tau_.allFinite()) {
          trip(kTripNonFinite);
          break;
        }
        if (pub_policy_->trylock()) {
          std::vector<double>& m = pub_policy_->msg_.data;
          int k = 0;
          m[k++] = clock_;
          m[k++] = core_.reach();
          m[k++] = core_.step();
          m[k++] = core_.go();
          for (int i = 0; i < 28; ++i) m[k++] = core_.last_obs()(i);
          for (int i = 0; i < 7; ++i) m[k++] = core_.u()(i);
          for (int i = 0; i < 7; ++i) m[k++] = policy_tau_(i);
          pub_policy_->unlockAndPublish();
        }
      }
      ++policy_ticks_;
      const Field& f = *field_buffer_.readFromRT();
      force << f.v[0] * v_handle.x() + f.v[1] * v_handle.y() + f.v[4],
               f.v[2] * v_handle.x() + f.v[3] * v_handle.y() + f.v[5], f.v[6];
      if (force.norm() > field_force_max_) force *= field_force_max_ / force.norm();
      const double ramp = ramp_time_ > 0.0 ? std::min(1.0, mode_time_ / ramp_time_) : 1.0;
      const Vector7d policy_total = (policy_tau_ - policy_damping_.cwiseProduct(dq)).cwiseMax(-tau_cap_).cwiseMin(tau_cap_);
      tau = ramp * (policy_total + J_handle.transpose() * force);
      break;
    }
    case Mode::kTripped:
      break;
  }
  if (mode_ == Mode::kTripped) tau = -d_trip_.cwiseProduct(dq);
  if (!tau.allFinite()) {
    trip(kTripNonFinite);
    tau = -d_trip_.cwiseProduct(dq);
  }
  const Vector7d tau_sent = saturateTorqueRate(tau, tau_J_d);
  for (size_t i = 0; i < 7; ++i) joint_handles_[i].setCommand(tau_sent(i));

  if (mode_ == Mode::kRunning) fast_until_ = clock_ + 1.0;              // keep logging 1 s past the episode
  if (clock_ < fast_until_ && pub_fast_->trylock()) {
    const std::array<double, 7> gravity = model_handle_->getGravity();
    std::vector<double>& m = pub_fast_->msg_.data;
    int k = 0;
    m[k++] = clock_;
    m[k++] = static_cast<double>(mode_ == Mode::kHold ? kStatusHold : mode_ == Mode::kHoming ? kStatusHoming
                                 : mode_ == Mode::kRunning ? kStatusRunning : kStatusTripped);
    for (int i = 0; i < 7; ++i) m[k++] = q(i);
    for (int i = 0; i < 7; ++i) m[k++] = dq(i);
    for (int i = 0; i < 7; ++i) m[k++] = rs.tau_J[i];
    for (int i = 0; i < 7; ++i) m[k++] = tau_J_d(i);
    for (int i = 0; i < 7; ++i) m[k++] = tau_sent(i);
    for (int i = 0; i < 7; ++i) m[k++] = gravity[i];
    for (int i = 0; i < 7; ++i) m[k++] = rs.tau_ext_hat_filtered[i];
    pub_fast_->unlockAndPublish();
  }

  if (tick_++ % 10 == 0 && pub_telemetry_->trylock()) {
    std::vector<double>& m = pub_telemetry_->msg_.data;
    int k = 0;
    m[k++] = time.toSec();
    m[k++] = static_cast<double>(mode_ == Mode::kHold ? kStatusHold : mode_ == Mode::kHoming ? kStatusHoming
                                 : mode_ == Mode::kRunning ? kStatusRunning : kStatusTripped);
    m[k++] = last_event_;
    m[k++] = core_.reach();
    m[k++] = core_.step();
    m[k++] = core_.go();
    for (int i = 0; i < 3; ++i) m[k++] = core_.target()(i);
    for (int i = 0; i < 3; ++i) m[k++] = handle(i);
    for (int i = 0; i < 3; ++i) m[k++] = v_handle(i);
    for (int i = 0; i < 7; ++i) m[k++] = q(i);
    for (int i = 0; i < 7; ++i) m[k++] = dq(i);
    for (int i = 0; i < 7; ++i) m[k++] = tau(i);
    for (int i = 0; i < 7; ++i) m[k++] = core_.u()(i);
    for (int i = 0; i < 3; ++i) m[k++] = force(i);
    pub_telemetry_->unlockAndPublish();
  }
}

Vector7d MotorCortexController::saturateTorqueRate(const Vector7d& tau_d, const Vector7d& tau_J_d) const {
  Vector7d out;
  for (int i = 0; i < 7; ++i) {
    out(i) = tau_J_d(i) + std::max(std::min(tau_d(i) - tau_J_d(i), delta_tau_max_), -delta_tau_max_);
  }
  return out;
}

void MotorCortexController::enterHold(const Vector7d& q) {
  q_hold_ = q;
  mode_ = Mode::kHold;
  mode_time_ = 0.0;
}

void MotorCortexController::trip(int reason) {
  if (mode_ == Mode::kTripped) return;
  mode_ = Mode::kTripped;
  mode_time_ = 0.0;
  fast_until_ = clock_ + 1.0;                                           // log the second after the trip
  publishStatus(reason);
}

void MotorCortexController::publishStatus(int code) {
  last_event_ = code;
  if (pub_status_->trylock()) {
    pub_status_->msg_.data = code;
    pub_status_->unlockAndPublish();
  }
}

void MotorCortexController::commandCallback(const std_msgs::StringConstPtr& msg) {
  std::istringstream in(msg->data);
  std::string word;
  in >> word;
  if (word == "home") {
    pending_command_ = kHome;
  } else if (word == "stop") {
    pending_command_ = kStop;
  } else if (word == "reset") {
    pending_command_ = kReset;
  } else if (word == "damping") {
    double k = -1.0;
    in >> k;
    if (!(k >= 0.0 && k <= 30.0)) {
      ROS_WARN("MotorCortexController: damping needs 0-30 N m s/rad; ignoring");
      return;
    }
    damping_request_ = k;
    pending_command_ = kDamping;
  } else if (word == "pulse") {
    int joint = 0;
    double amp = 0.0, ms = 0.0;
    in >> joint >> amp >> ms;
    if (joint < 1 || joint > 7 || !std::isfinite(amp) || std::abs(amp) > 5.0 || !(ms >= 10.0 && ms <= 200.0)) {
      ROS_WARN("MotorCortexController: pulse needs joint 1-7, |N m| <= 5, 10-200 ms; ignoring");
      return;
    }
    Pulse p;
    p.joint = joint - 1;
    p.amp = amp;
    p.duration = ms / 1000.0;
    pulse_buffer_.writeFromNonRT(p);
    pending_command_ = kPulse;
  } else if (word == "start") {
    int n_out = 6;
    long seed = static_cast<long>(ros::Time::now().toNSec() % 1000003);
    in >> n_out >> seed;
    n_out = std::max(1, std::min(n_out, MotorCortexEpisode::kMaxReaches / 2));
    EpisodePlan plan;
    plan.n_out = n_out;
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::array<int, 8> order{0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < n_out; ++i) {
      if (i % 8 == 0) std::shuffle(order.begin(), order.end(), rng);
      plan.angle[i] = order[i % 8] * 2.0 * M_PI / 8.0;
    }
    plan_buffer_.writeFromNonRT(plan);
    pending_command_ = kStart;
    ROS_INFO("MotorCortexController: start requested, %d outward reaches, seed %ld", n_out, seed);
  } else {
    ROS_WARN_STREAM("MotorCortexController: unknown command '" << msg->data << "'");
  }
}

void MotorCortexController::fieldCallback(const std_msgs::Float64MultiArrayConstPtr& msg) {
  if (msg->data.size() != 7) {
    ROS_WARN("MotorCortexController: field needs 7 values (B 2x2 row-major [N s/m], c 3 [N]); ignoring");
    return;
  }
  Field f;
  for (int i = 0; i < 7; ++i) {
    f.v[i] = msg->data[i];
    const double limit = i < 4 ? 100.0 : 25.0;
    if (!std::isfinite(f.v[i]) || std::abs(f.v[i]) > limit) {
      ROS_WARN("MotorCortexController: field value %d out of range (|B| <= 100 N s/m, |c| <= 25 N); ignoring", i);
      return;
    }
  }
  field_buffer_.writeFromNonRT(f);
}

}  // namespace panda_controllers

PLUGINLIB_EXPORT_CLASS(panda_controllers::MotorCortexController, controller_interface::ControllerBase)
