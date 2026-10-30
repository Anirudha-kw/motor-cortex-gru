// ROS-free core of the motor-cortex controller: the in-context GRU policy (forcefield_gru/meta.py, retrained for
// the real arm by s2r.py), the centre-out reach sequencer of the training probes, and the policy's 10 ms sensing
// delay.  The controller calls tick() once per 10 ms with the robot's current sensing; tick() feeds the policy the
// sensing of the PREVIOUS tick (the delay it was trained with -- the robot itself senses in 1-2 ms, so the delay is
// imposed here, exactly) and returns the joint torque to hold for the next 10 ms.  Everything is preallocated in
// load(); begin() and tick() do not allocate.
//
// Observation (meta.rollout): [(target - home) * 5, go, (handle - home) * 5, q - q_home, dq * 0.2, u_prev], 28 values.
// Output: u = tanh(W_out' h + b_out) in (-1, 1); torque = u .* tau_max [N m], on top of Franka's gravity compensation.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace panda_controllers {

using Vector7d = Eigen::Matrix<double, 7, 1>;

struct MotorCortexSense {
  Eigen::Vector3d handle = Eigen::Vector3d::Zero();   // handle position in the base frame [m]
  Vector7d q = Vector7d::Zero();                      // joint angles [rad]
  Vector7d dq = Vector7d::Zero();                     // joint velocities [rad/s]
};

struct MotorCortexEpisode {
  static constexpr int kMaxReaches = 32;
  int n_reaches = 0;
  std::array<Eigen::Vector3d, kMaxReaches> targets{};
};

class MotorCortexCore {
 public:
  static constexpr int kReachSteps = 70;   // 0.7 s per reach (meta.REACH_STEPS)
  static constexpr int kGoSeen = 11;       // go cue at step 10, seen one step later (meta.GO_TEST + DELAY)

  bool load(const std::string& path, std::string* error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(error, "cannot open " + path);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "MCGRU001", 8) != 0) return fail(error, "not an MCGRU001 file: " + path);
    int32_t dims[3];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    n_in_ = dims[0];
    h_dim_ = dims[1];
    const int n_out = dims[2];
    if (!f || n_in_ != 28 || n_out != 7 || h_dim_ <= 0 || h_dim_ > 4096) return fail(error, "unexpected dimensions");
    auto read = [&f](Eigen::Ref<Eigen::MatrixXd> m, int rows, int cols) {   // row-major on disk
      std::vector<double> buf(static_cast<size_t>(rows) * cols);
      f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(double)));
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) m(r, c) = buf[static_cast<size_t>(r) * cols + c];
    };
    const int H3 = 3 * h_dim_;
    W_x_.resize(n_in_, H3);  b_x_.resize(H3);
    W_h_.resize(h_dim_, H3); b_h_.resize(H3);
    W_out_.resize(h_dim_, 7); b_out_.resize(7);
    read(W_x_, n_in_, H3);   read(b_x_, H3, 1);
    read(W_h_, h_dim_, H3);  read(b_h_, H3, 1);
    read(W_out_, h_dim_, 7); read(b_out_, 7, 1);
    Eigen::Matrix<double, 7, 1> qh, tm;
    Eigen::Vector3d hs;
    read(qh, 7, 1); read(tm, 7, 1); read(hs, 3, 1);
    q_home_ = qh; tau_max_ = tm; home_sim_ = hs;
    char end[3];
    f.read(end, 3);
    if (!f || std::memcmp(end, "END", 3) != 0) return fail(error, "truncated file: " + path);
    if (!W_x_.allFinite() || !W_h_.allFinite() || !W_out_.allFinite()) return fail(error, "non-finite weights");
    x_.resize(n_in_); gx_.resize(H3); gh_.resize(H3); h_.resize(h_dim_); hu_.resize(7);
    h_.setZero(); u_prev_.setZero(); torque_.setZero();
    loaded_ = true;
    return true;
  }

  // Start an episode: zero the GRU state and efference copy; the delay line starts filled with `now`.
  void begin(const MotorCortexEpisode& episode, const Eigen::Vector3d& home, const MotorCortexSense& now) {
    episode_ = episode;
    home_ = home;
    h_.setZero();
    u_prev_.setZero();
    torque_.setZero();
    previous_ = now;
    reach_ = 0;
    step_ = 0;
    running_ = episode.n_reaches > 0;
  }

  // One 10 ms period.  Returns the torque [N m] to hold until the next tick (zero once the episode is over).
  const Vector7d& tick(const MotorCortexSense& now) {
    if (!running_) {
      torque_.setZero();
      return torque_;
    }
    target_ = episode_.targets[reach_];
    go_ = step_ >= kGoSeen ? 1.0 : 0.0;
    const MotorCortexSense& seen = previous_;                     // 10 ms old
    x_.segment<3>(0) = (target_ - home_) * 5.0;
    x_(3) = go_;
    x_.segment<3>(4) = (seen.handle - home_) * 5.0;
    x_.segment<7>(7) = seen.q - q_home_;
    x_.segment<7>(14) = seen.dq * 0.2;
    x_.segment<7>(21) = u_prev_;
    gx_.noalias() = W_x_.transpose() * x_;
    gx_ += b_x_;
    gh_.noalias() = W_h_.transpose() * h_;
    gh_ += b_h_;
    const int H = h_dim_;
    for (int j = 0; j < H; ++j) {
      const double r = sigmoid(gx_(j) + gh_(j));
      const double z = sigmoid(gx_(H + j) + gh_(H + j));
      const double n = std::tanh(gx_(2 * H + j) + r * gh_(2 * H + j));
      h_(j) = (1.0 - z) * n + z * h_(j);
    }
    hu_.noalias() = W_out_.transpose() * h_;
    for (int i = 0; i < 7; ++i) u_prev_(i) = std::tanh(hu_(i) + b_out_(i));
    torque_ = u_prev_.cwiseProduct(tau_max_);
    previous_ = now;
    if (++step_ == kReachSteps) {
      step_ = 0;
      if (++reach_ == episode_.n_reaches) running_ = false;
    }
    return torque_;
  }

  bool loaded() const { return loaded_; }
  bool running() const { return running_; }
  int reach() const { return reach_; }
  int step() const { return step_; }
  double go() const { return go_; }
  int hidden() const { return h_dim_; }
  const Eigen::Vector3d& target() const { return target_; }
  const Vector7d& u() const { return u_prev_; }
  const Eigen::VectorXd& last_obs() const { return x_; }    // the 28 inputs of the last tick
  const Vector7d& q_home() const { return q_home_; }
  const Vector7d& tau_max() const { return tau_max_; }
  const Eigen::Vector3d& home_sim() const { return home_sim_; }
  bool finite() const { return h_.allFinite() && u_prev_.allFinite(); }

 private:
  static double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }
  static bool fail(std::string* error, const std::string& what) {
    if (error) *error = what;
    return false;
  }

  bool loaded_ = false;
  int n_in_ = 0, h_dim_ = 0;
  Eigen::MatrixXd W_x_, W_h_, W_out_;
  Eigen::VectorXd b_x_, b_h_, b_out_, x_, gx_, gh_, h_, hu_;
  Vector7d q_home_ = Vector7d::Zero(), tau_max_ = Vector7d::Zero(), u_prev_ = Vector7d::Zero(),
           torque_ = Vector7d::Zero();
  Eigen::Vector3d home_sim_ = Eigen::Vector3d::Zero(), home_ = Eigen::Vector3d::Zero(),
                  target_ = Eigen::Vector3d::Zero();
  MotorCortexEpisode episode_;
  MotorCortexSense previous_;
  int reach_ = 0, step_ = 0;
  double go_ = 0.0;
  bool running_ = false;
};

}  // namespace panda_controllers
