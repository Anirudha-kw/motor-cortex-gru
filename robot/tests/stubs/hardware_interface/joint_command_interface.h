#pragma once
#include <stdexcept>
#include <string>
namespace hardware_interface {
struct HardwareInterfaceException : std::runtime_error { using std::runtime_error::runtime_error; };
struct JointHandle { void setCommand(double) {} };
struct EffortJointInterface { JointHandle getHandle(const std::string&) { return JointHandle(); } };
}
