#include "OffloadUtils.hpp"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

// The static 'Wrapper' instance ensures olInit() is called once at program
// startup and olShutDown() is called once at program termination.
struct OffloadInitWrapper {
  OffloadInitWrapper() { OLS_CHECK(olInit()); }
  ~OffloadInitWrapper() { OLS_CHECK(olShutDown()); }
};
static OffloadInitWrapper Wrapper{};

const std::vector<ols::Device> &ols::getDevices() {
  // Thread-safe initialization of a static local variable
  static std::vector<ols::Device> Devices = []() -> std::vector<ols::Device> {
    std::vector<ols::Device> TempDevices{};

    auto ResultFromIterate = olIterateDevices(
        [](ol_device_handle_t DeviceHandle, void *Data) {
          ol_platform_handle_t PlatformHandle = nullptr;
          OLS_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                                    sizeof(ol_platform_handle_t),
                                    &PlatformHandle));
          ol_platform_backend_t Backend;
          OLS_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_BACKEND,
                                      sizeof(Backend), &Backend));

          bool IsHost = Backend == OL_PLATFORM_BACKEND_HOST;
          bool IsCUDA = Backend == OL_PLATFORM_BACKEND_CUDA;
          bool IsAMDGPU = Backend == OL_PLATFORM_BACKEND_AMDGPU;

          static_cast<std::vector<ols::Device> *>(Data)->push_back(
              {DeviceHandle, IsHost, IsCUDA, IsAMDGPU});

          return true;
        },
        &TempDevices);

    OLS_CHECK(ResultFromIterate);

    return TempDevices;
  }();

  return Devices;
}

const ols::Device &ols::getHostDevice() {
  // Thread-safe initialization of a static local variable
  static const ols::Device &HostDeviceRef = []() -> const ols::Device & {
    const auto &Devices = ols::getDevices();

    for (const auto &CurrentDevice : ols::getDevices()) {
      if (CurrentDevice.IsHost) {
        return CurrentDevice;
      }
    }

    std::cerr << "FATAL ERROR: In function " << __func__ << " (" << __FILE__
              << ":" << __LINE__ << "): "
              << "The host device was not found" << '\n';
    std::exit(EXIT_FAILURE);
  }();

  return HostDeviceRef;
}

std::string getDeviceInfoAsString(const ols::Device &TargetDevice,
                                  ol_device_info_t PropName) {
  assert(TargetDevice.Handle != nullptr &&
         "getDeviceInfoAsString called with a null TargetDevice.Handle.");

  assert((PropName == OL_DEVICE_INFO_NAME ||
          PropName == OL_DEVICE_INFO_VENDOR ||
          PropName == OL_DEVICE_INFO_DRIVER_VERSION) &&
         "Invalid PropName passed to getDeviceInfoAsString.");

  size_t PropSize = 0;
  OLS_CHECK(olGetDeviceInfoSize(TargetDevice.Handle, PropName, &PropSize));

  if (PropSize == 0) {
    return "";
  }

  std::string PropValue(PropSize, '\0');
  OLS_CHECK(olGetDeviceInfo(TargetDevice.Handle, PropName, PropSize,
                            PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}

ols::DeviceInfo ols::getDeviceInfo(const Device &TargetDevice) {
  std::string Name;
  std::string Vendor;
  std::string DriverVersion;

  if (TargetDevice.IsHost) {
    Name = "Host";
  } else {
    Name = getDeviceInfoAsString(TargetDevice, OL_DEVICE_INFO_NAME);
    Vendor = getDeviceInfoAsString(TargetDevice, OL_DEVICE_INFO_VENDOR);
    DriverVersion =
        getDeviceInfoAsString(TargetDevice, OL_DEVICE_INFO_DRIVER_VERSION);
  }

  ol_platform_handle_t Platform = nullptr;
  return ols::DeviceInfo({Name, Vendor, DriverVersion});
}

const std::string DeviceBinsDirectory = DEVICE_CODE_PATH;

bool ols::loadDeviceBinary(const std::string &BinaryName,
                           const ols::Device &TargetDevice,
                           std::vector<char> &BinaryOut) {

  ol_platform_handle_t Platform = nullptr;
  OLS_CHECK(olGetDeviceInfo(TargetDevice.Handle, OL_DEVICE_INFO_PLATFORM,
                            sizeof(ol_platform_handle_t), &Platform));

  ol_platform_backend_t Backend;
  OLS_CHECK(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                              sizeof(Backend), &Backend));

  std::string FileExtension;
  switch (Backend) {
  case OL_PLATFORM_BACKEND_CUDA:
    FileExtension = ".nvptx64.bin";
    break;
  default:
    // TODO: Add support for AMDGPU and host CPU
    std::cerr << "Unsupported backend for a device binary\n";
    return false;
  }

  namespace fs = std::filesystem;
  fs::path BinaryPath =
      fs::path(DeviceBinsDirectory) / (BinaryName + FileExtension);

  // Open the device binary in *binary* mode and start *at end* so we can
  // query its size with tellg() before reading.
  std::ifstream BinaryFile(BinaryPath, std::ios::binary | std::ios::ate);
  if (!BinaryFile) {
    std::cerr << "Failed to open the device binary: " << BinaryPath << '\n';
    return false;
  }

  std::streamsize BinarySize = BinaryFile.tellg();
  BinaryOut.resize(BinarySize);
  BinaryFile.seekg(0, std::ios::beg);
  if (!BinaryFile.read(BinaryOut.data(), BinarySize)) {
    std::cerr << "Failed to read the device binary: " << BinaryPath << '\n';
    return false;
  }

  return true;
}