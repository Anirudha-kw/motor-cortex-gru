#pragma once
#include <memory>
#include <string>
#include <vector>
#include "harness_world.h"
namespace ros {
struct TransportHints { TransportHints& reliable() { return *this; } TransportHints& tcpNoDelay() { return *this; } };
struct Subscriber {};
struct NodeHandle {
  bool getParam(const std::string& n, std::string& v) const { auto i = harness::str_params.find(n); if (i == harness::str_params.end()) return false; v = i->second; return true; }
  bool getParam(const std::string& n, std::vector<double>& v) const { auto i = harness::vec_params.find(n); if (i == harness::vec_params.end()) return false; v = i->second; return true; }
  bool getParam(const std::string& n, std::vector<std::string>& v) const { auto i = harness::vstr_params.find(n); if (i == harness::vstr_params.end()) return false; v = i->second; return true; }
  void param(const std::string& n, double& out, const double& def) const { auto i = harness::num_params.find(n); out = i == harness::num_params.end() ? def : i->second; }
  template <class M, class C> Subscriber subscribe(const std::string&, int, void (C::*)(const std::shared_ptr<const M>&), C*,
                                                    const TransportHints& = TransportHints()) { return Subscriber(); }
};
}
