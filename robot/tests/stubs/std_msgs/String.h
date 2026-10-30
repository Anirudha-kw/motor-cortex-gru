#pragma once
#include <memory>
#include <string>
#include <vector>
namespace std_msgs { struct String { std::string data{}; }; using StringConstPtr = std::shared_ptr<const String>; }
