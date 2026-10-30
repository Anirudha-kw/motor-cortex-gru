#pragma once
#include <stdexcept>
#include <string>
#include "harness_world.h"
namespace hardware_interface {
struct HardwareInterfaceException : std::runtime_error { using std::runtime_error::runtime_error; };
struct JointHandle { int i = 0; void setCommand(double v) { harness::command[i] = v; } };
struct EffortJointInterface { JointHandle getHandle(const std::string& name) { JointHandle h; h.i = name.back() - '1'; return h; } };
}
