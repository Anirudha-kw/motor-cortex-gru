#pragma once
namespace hardware_interface { struct RobotHW { template <class T> T* get() { static T instance; return &instance; } }; }
