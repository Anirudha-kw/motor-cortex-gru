#pragma once
#include <string>
#include <ros/node_handle.h>
namespace realtime_tools { template <class M> struct RealtimePublisher {
  RealtimePublisher(const ros::NodeHandle&, const std::string&, int, bool = false) {}
  bool trylock() { return true; } void unlockAndPublish() {} M msg_; }; }
