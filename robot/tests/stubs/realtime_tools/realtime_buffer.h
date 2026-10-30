#pragma once
namespace realtime_tools { template <class T> struct RealtimeBuffer { T v{}; void writeFromNonRT(const T& d) { v = d; } T* readFromRT() { return &v; } }; }
