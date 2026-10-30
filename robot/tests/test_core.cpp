// Replays a reference trace (make_trace.py) through MotorCortexCore and checks every tick's torque, target and go
// cue against the Python policy: the GRU, the observation, the sequencer and the 10 ms delay together.
//   g++ -O2 -std=c++14 -I/usr/include/eigen3 -I../include test_core.cpp -o test_core
//   ./test_core ../weights/motor_cortex_s2rR_s1.mcgru trace_s2rR_s1.txt
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <panda/motor_cortex_core.h>

using panda_controllers::MotorCortexCore;
using panda_controllers::MotorCortexEpisode;
using panda_controllers::MotorCortexSense;

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: test_core weights.mcgru trace.txt\n");
    return 2;
  }
  MotorCortexCore core;
  std::string error;
  if (!core.load(argv[1], &error)) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  std::ifstream f(argv[2]);
  std::string line;
  std::getline(f, line);
  std::istringstream head(line);
  Eigen::Vector3d home;
  panda_controllers::Vector7d q0;
  for (int i = 0; i < 3; ++i) head >> home(i);
  for (int i = 0; i < 7; ++i) head >> q0(i);
  struct Row { MotorCortexSense now; Eigen::Vector3d target; double go; panda_controllers::Vector7d tau; };
  std::vector<Row> rows;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    Row r;
    for (int i = 0; i < 3; ++i) s >> r.now.handle(i);
    for (int i = 0; i < 7; ++i) s >> r.now.q(i);
    for (int i = 0; i < 7; ++i) s >> r.now.dq(i);
    for (int i = 0; i < 3; ++i) s >> r.target(i);
    s >> r.go;
    for (int i = 0; i < 7; ++i) s >> r.tau(i);
    rows.push_back(r);
  }
  MotorCortexEpisode ep;
  ep.n_reaches = static_cast<int>(rows.size()) / MotorCortexCore::kReachSteps;
  for (int r = 0; r < ep.n_reaches; ++r) ep.targets[r] = rows[r * MotorCortexCore::kReachSteps].target;
  MotorCortexSense start;
  start.handle = home;
  start.q = q0;
  core.begin(ep, home, start);
  double worst_tau = 0.0, worst_target = 0.0;
  int go_mismatch = 0;
  for (size_t k = 0; k < rows.size(); ++k) {
    const auto tau = core.tick(rows[k].now);
    worst_tau = std::max(worst_tau, (tau - rows[k].tau).cwiseAbs().maxCoeff());
    worst_target = std::max(worst_target, (core.target() - rows[k].target).cwiseAbs().maxCoeff());
    go_mismatch += core.go() != rows[k].go;
  }
  std::printf("%zu ticks, %d reaches, hidden %d: max |torque - python| %.3g N m, max |target| diff %.3g m, "
              "go mismatches %d, episode over %s\n", rows.size(), ep.n_reaches, core.hidden(), worst_tau,
              worst_target, go_mismatch, core.running() ? "no" : "yes");
  const bool ok = worst_tau < 1e-6 && worst_target < 1e-12 && go_mismatch == 0 && !core.running();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
