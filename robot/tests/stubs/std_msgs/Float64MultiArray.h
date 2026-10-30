#pragma once
#include <memory>
#include <string>
#include <vector>
namespace std_msgs { struct Float64MultiArray { std::vector<double> data{}; }; using Float64MultiArrayConstPtr = std::shared_ptr<const Float64MultiArray>; }
