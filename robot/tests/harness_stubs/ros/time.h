#pragma once
#include <cstdint>
namespace ros { struct Time { double s = 0; static Time now() { return Time(); } double toSec() const { return s; } uint64_t toNSec() const { return 0; } };
struct Duration { double s = 0.001; double toSec() const { return s; } }; }
