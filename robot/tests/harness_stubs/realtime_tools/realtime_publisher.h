#pragma once
#include <string>
#include <ros/node_handle.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>
namespace realtime_tools {
inline void record(const std_msgs::Int32& m) { harness::status.push_back(m.data); }
inline void record(const std_msgs::Float64MultiArray& m, const std::string& topic) {
  (topic == "policy_log" ? harness::policy_log : topic == "fast_log" ? harness::fast_log : harness::telemetry).push_back(m.data);
}
inline void record(const std_msgs::Int32& m, const std::string&) { record(m); }
template <class M> struct RealtimePublisher {
  RealtimePublisher(const ros::NodeHandle&, const std::string& topic, int, bool = false) : topic_(topic) {}
  bool trylock() { return true; } void unlockAndPublish() { record(msg_, topic_); } M msg_; std::string topic_; };
}
