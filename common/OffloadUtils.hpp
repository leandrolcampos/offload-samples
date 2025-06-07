#pragma once

#include <cstdlib>
#include <iostream>
#include <offload/OffloadAPI.h>
#include <string>
#include <vector>

namespace ols {

namespace internal {

// OL_CHECK failure handler. Separated from the macro to ensure this header
// explicitly includes <cstdlib> and <iostream> for its expansion
[[noreturn]] inline void checkFailureHandler(const char *ResultExpr,
                                             ol_result_t Result,
                                             const char *File, int Line,
                                             const char *FuncName) {
  std::cerr << "OL_CHECK FAILED: (" << ResultExpr << ") != OL_SUCCESS\n"
            << "  Error Code: " << Result->Code << '\n'
            << "  Details: "
            << (Result->Details ? Result->Details : "No details provided")
            << '\n'
            << "  Location: " << File << ":" << Line << '\n'
            << "  Function: " << FuncName << '\n';
  std::exit(EXIT_FAILURE);
}

// FATAL_ERROR failure handler
[[noreturn]] inline void fatalErrorHandler(const std::string &Message,
                                           const char *File, int Line,
                                           const char *FuncName) {
  std::cerr << "FATAL ERROR: " << Message << '\n'
            << "  Location: " << File << ":" << Line << '\n'
            << "  Function: " << FuncName << '\n';
  std::exit(EXIT_FAILURE);
}

} // namespace internal

#define OL_CHECK(ResultExpr)                                                   \
  do {                                                                         \
    ol_result_t Result = (ResultExpr);                                         \
    if (Result != OL_SUCCESS) {                                                \
      ols::internal::checkFailureHandler(#ResultExpr, Result, __FILE__,        \
                                         __LINE__, __func__);                  \
    }                                                                          \
  } while (false)

#define FATAL_ERROR(Message)                                                   \
  ols::internal::fatalErrorHandler(Message, __FILE__, __LINE__, __func__)

struct Device {
  ol_device_handle_t Handle = nullptr;
  bool IsCUDA = false;
  bool IsAMDGPU = false;
};

struct DeviceInfo {
  std::string Name;
  std::string Vendor;
  std::string DriverVersion;
};

const std::vector<Device> &getDevices();

ol_device_handle_t getHostHandle();

DeviceInfo getDeviceInfo(const Device &TargetDevice);

void loadDeviceBinary(const std::string &BinaryName, const Device &TargetDevice,
                      std::vector<char> &BinaryOut);

} // namespace ols