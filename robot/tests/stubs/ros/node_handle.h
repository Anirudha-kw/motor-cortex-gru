#pragma once
#include <memory>
#include <string>
#include <vector>
namespace ros {
struct TransportHints { TransportHints& reliable() { return *this; } TransportHints& tcpNoDelay() { return *this; } };
struct Subscriber {};
struct NodeHandle {
  template <class T> bool getParam(const std::string&, T&) const { return true; }
  template <class T> void param(const std::string&, T& out, const T& def) const { out = def; }
  template <class M, class C> Subscriber subscribe(const std::string&, int, void (C::*)(const std::shared_ptr<const M>&), C*,
                                                    const TransportHints& = TransportHints()) { return Subscriber(); }
};
}
