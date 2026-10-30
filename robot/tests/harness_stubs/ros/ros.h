#pragma once
#include <iostream>
#include <ros/node_handle.h>
#include <ros/time.h>
#define ROS_INFO(...) ((void)0)
#define ROS_WARN(...) ((void)0)
#define ROS_ERROR(...) ((void)0)
#define ROS_INFO_STREAM(x) (std::cerr << x << std::endl)
#define ROS_WARN_STREAM(x) (std::cerr << x << std::endl)
#define ROS_ERROR_STREAM(x) (std::cerr << x << std::endl)
