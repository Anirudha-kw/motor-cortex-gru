#pragma once
#include <controller_interface/controller_base.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
namespace controller_interface {
template <class... T> struct MultiInterfaceController : ControllerBase {
  virtual bool init(hardware_interface::RobotHW*, ros::NodeHandle&) = 0;
  virtual void starting(const ros::Time&) {}
  virtual void update(const ros::Time&, const ros::Duration&) = 0;
};
}
