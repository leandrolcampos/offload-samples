#include "DeviceContext.hpp"
#include "Support.hpp"
#include <llvm/Support/raw_ostream.h>

// The static 'Wrapper' instance ensures olInit() is called once at program
// startup and olShutDown() is called once at program termination
struct OffloadInitWrapper {
  OffloadInitWrapper() { OL_CHECK(olInit()); }
  ~OffloadInitWrapper() { OL_CHECK(olShutDown()); }
};
static OffloadInitWrapper Wrapper{};

const static std::vector<ol_device_handle_t> getDevices() {
  // Thread-safe initialization of a static local variable
  static std::vector<ol_device_handle_t> Devices =
      []() -> std::vector<ol_device_handle_t> {
    std::vector<ol_device_handle_t> TempDevices;

    // Discovery every device that is not the host
    const auto *const ResultFromIterate = olIterateDevices(
        [](ol_device_handle_t DeviceHandle, void *Data) {
          ol_platform_handle_t PlatformHandle = nullptr;
          OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                                   sizeof(ol_platform_handle_t),
                                   &PlatformHandle));
          ol_platform_backend_t Backend;
          OL_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_BACKEND,
                                     sizeof(Backend), &Backend));

          if (Backend != OL_PLATFORM_BACKEND_HOST) {
            static_cast<std::vector<ol_device_handle_t> *>(Data)->push_back(
                DeviceHandle);
          }
          return true;
        },
        &TempDevices);

    OL_CHECK(ResultFromIterate);

    return TempDevices;
  }();

  return Devices;
}

namespace testing {

size_t countDevices() { return getDevices().size(); }

DeviceContext::DeviceContext(size_t DeviceId)
    : DeviceId(DeviceId), DeviceHandle(nullptr) {
  const auto &Devices = getDevices();

  if (DeviceId >= Devices.size()) {
    FATAL_ERROR("Invalid DeviceId: " + std::to_string(DeviceId) +
                ", but only " + std::to_string(Devices.size()) +
                " devices are available.");
  }

  DeviceHandle = Devices[DeviceId];
}

[[nodiscard]] std::shared_ptr<DeviceImage>
DeviceContext::loadBinary(const std::string &Directory,
                          const std::string &BinaryName,
                          const std::string &Extension) const {
  std::string FullPath = Directory + "/" + BinaryName + Extension;

  // For simplicity, this implementation intentionally reads the binary from
  // disk on every call.
  //
  // Other use cases could benefit from a global, thread-safe cache (likely
  // using std::weak_ptr) to avoid redundant file I/O and GPU program creation.

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> FileOrErr =
      llvm::MemoryBuffer::getFile(FullPath);
  if (std::error_code ErrorCode = FileOrErr.getError()) {
    FATAL_ERROR("Failed to read device binary file '" + FullPath +
                "': " + ErrorCode.message());
  }
  std::unique_ptr<llvm::MemoryBuffer> &BinaryData = *FileOrErr;

  ol_program_handle_t ProgramHandle = nullptr;
  OL_CHECK(olCreateProgram(DeviceHandle, BinaryData->getBufferStart(),
                           BinaryData->getBufferSize(), &ProgramHandle));

  return std::shared_ptr<DeviceImage>(
      new DeviceImage(DeviceHandle, ProgramHandle));
}

[[nodiscard]] std::shared_ptr<DeviceImage>
DeviceContext::loadBinary(const std::string &Directory,
                          const std::string &BinaryName) const {
  std::string Extension;
  ol_platform_backend_t Backend = getBackend();
  if (Backend == OL_PLATFORM_BACKEND_AMDGPU) {
    Extension = ".amdgpu.bin";
  } else if (Backend == OL_PLATFORM_BACKEND_CUDA) {
    Extension = ".nvptx64.bin";
  } else {
    FATAL_ERROR("Unsupported backend to infer binary extension");
  }
  return loadBinary(Directory, BinaryName, Extension);
}

std::string DeviceContext::getName() const {
  size_t PropSize = 0;
  OL_CHECK(olGetDeviceInfoSize(DeviceHandle, OL_DEVICE_INFO_NAME, &PropSize));

  if (PropSize == 0) {
    return "";
  }

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_NAME, PropSize,
                           PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}

std::string DeviceContext::getPlatform() const {
  ol_platform_handle_t PlatformHandle = nullptr;
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                           sizeof(ol_platform_handle_t), &PlatformHandle));

  size_t PropSize = 0;
  OL_CHECK(
      olGetPlatformInfoSize(PlatformHandle, OL_PLATFORM_INFO_NAME, &PropSize));

  if (PropSize == 0) {
    return "";
  }

  std::string PropValue(PropSize, '\0');
  OL_CHECK(olGetPlatformInfo(PlatformHandle, OL_PLATFORM_INFO_NAME, PropSize,
                             PropValue.data()));
  PropValue.pop_back(); // Remove the null terminator

  return PropValue;
}

ol_platform_backend_t DeviceContext::getBackend() const {
  ol_platform_handle_t Platform;
  OL_CHECK(olGetDeviceInfo(DeviceHandle, OL_DEVICE_INFO_PLATFORM,
                           sizeof(Platform), &Platform));
  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  OL_CHECK(olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                             sizeof(Backend), &Backend));
  return Backend;
}

} // namespace testing