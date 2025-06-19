#pragma once

#include "Support.hpp"
#include "offload/OffloadAPI.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

#define OL_CHECK(ResultExpr)                                                   \
  do {                                                                         \
    ol_result_t Result = (ResultExpr);                                         \
    if (Result != OL_SUCCESS) {                                                \
      testing::internal::reportOffloadError(#ResultExpr, Result, __FILE__,     \
                                            __LINE__, __func__);               \
    }                                                                          \
  } while (false)

namespace testing {
namespace internal {

[[noreturn]] inline void reportOffloadError(const char *ResultExpr,
                                            ol_result_t Result,
                                            const char *File, int Line,
                                            const char *FuncName) {
  llvm::errs() << "--- OL_CHECK FAILED ---\n"
               << "Location: " << File << ":" << Line << "\n"
               << "Function: " << FuncName << "\n"
               << "Check: (" << ResultExpr << ") != OL_SUCCESS\n"
               << "  Error Code: " << Result->Code << "\n"
               << "  Details: "
               << (Result->Details ? Result->Details : "No details provided")
               << "\n\n";
  std::exit(EXIT_FAILURE);
}

} // namespace internal

size_t countDevices() noexcept;

class DeviceContext;

template <typename T> class [[nodiscard]] ManagedBuffer {
private:
  friend class DeviceContext;

  T *Address = nullptr;
  size_t Size = 0;

  ManagedBuffer(T *Address, size_t Size) : Address(Address), Size(Size) {}

public:
  ~ManagedBuffer() noexcept {
    if (Address) {
      OL_CHECK(olMemFree(Address));
    }
  }

  ManagedBuffer(const ManagedBuffer &) = delete;
  ManagedBuffer &operator=(const ManagedBuffer &) = delete;

  ManagedBuffer(ManagedBuffer &&Other) noexcept
      : Address(Other.Address), Size(Other.Size) {
    Other.Address = nullptr;
    Other.Size = 0;
  }

  ManagedBuffer &operator=(ManagedBuffer &&Other) noexcept {
    if (this == &Other)
      return *this;

    if (Address) {
      OL_CHECK(olMemFree(Address));
    }

    Address = Other.Address;
    Size = Other.Size;

    Other.Address = nullptr;
    Other.Size = 0;

    return *this;
  }

  T *data() noexcept { return Address; }

  const T *data() const noexcept { return Address; }

  [[nodiscard]] size_t getSize() const noexcept { return Size; }
};

class [[nodiscard]] DeviceImage {
private:
  friend class DeviceContext;

  ol_device_handle_t DeviceHandle = nullptr;
  ol_program_handle_t Handle = nullptr;

  DeviceImage(ol_device_handle_t DeviceHandle, ol_program_handle_t Handle)
      : DeviceHandle(DeviceHandle), Handle(Handle) {}

public:
  ~DeviceImage() noexcept {
    if (Handle) {
      OL_CHECK(olDestroyProgram(Handle));
    }
  }

  DeviceImage(const DeviceImage &) = delete;
  DeviceImage &operator=(const DeviceImage &) = delete;

  DeviceImage(DeviceImage &&Other) noexcept
      : DeviceHandle(Other.DeviceHandle), Handle(Other.Handle) {
    Other.DeviceHandle = nullptr;
    Other.Handle = nullptr;
  }

  DeviceImage &operator=(DeviceImage &&Other) noexcept {
    if (this == &Other)
      return *this;

    if (Handle) {
      OL_CHECK(olDestroyProgram(Handle));
    }

    DeviceHandle = Other.DeviceHandle;
    Handle = Other.Handle;

    Other.DeviceHandle = nullptr;
    Other.Handle = nullptr;

    return *this;
  }
};

template <typename KernelSignature> class [[nodiscard]] DeviceKernel {
private:
  friend class DeviceContext;

  std::shared_ptr<DeviceImage> Image;
  ol_kernel_handle_t Handle = nullptr;

  DeviceKernel(std::shared_ptr<DeviceImage> Image, ol_kernel_handle_t Kernel)
      : Image(Image), Handle(Kernel) {}

public:
  DeviceKernel() = delete;

  DeviceKernel(const DeviceKernel &) = default;
  DeviceKernel &operator=(const DeviceKernel &) = default;
  DeviceKernel(DeviceKernel &&) noexcept = default;
  DeviceKernel &operator=(DeviceKernel &&) noexcept = default;
};

class Dim {
private:
  friend class DeviceContext;

  uint32_t Data[3] = {1, 1, 1};

  constexpr operator ol_dimensions_t() const noexcept {
    return {Data[0], Data[1], Data[2]};
  }

public:
  constexpr Dim() = default;

  constexpr Dim(uint32_t X, uint32_t Y = 1, uint32_t Z = 1) : Data{X, Y, Z} {
    assert(X > 0 && Y > 0 && Z > 0 && "Dimensions must be positive");
  }

  constexpr Dim(std::initializer_list<uint32_t> Dimensions) {
    assert(Dimensions.size() <= 3 &&
           "The number of dimensions must be less than or equal to 3");

    const auto *It = Dimensions.begin();
    auto X = Data[0] = (Dimensions.size() >= 1) ? *It++ : 1;
    auto Y = Data[1] = (Dimensions.size() >= 2) ? *It++ : 1;
    auto Z = Data[2] = (Dimensions.size() >= 3) ? *It++ : 1;

    assert(X > 0 && Y > 0 && Z > 0 && "Dimensions must be positive");
  }

  constexpr uint32_t operator[](size_t Index) const noexcept {
    assert(Index < 3 && "Index is out of range");
    return Data[Index];
  }
};

class DeviceContext {
  // For simplicity, the current design of this class doesn't have support for
  // asynchronous operations and all types of memory allocation.
  //
  // Other use cases could benefit from operations like enqueued kernel launch
  // and enqueued memcpy, as well as device and host memory allocation.

private:
  size_t DeviceId;
  ol_device_handle_t DeviceHandle;

public:
  DeviceContext(size_t DeviceId = 0);

  template <typename T>
  ManagedBuffer<T> createManagedBuffer(size_t Size) const {
    T *Address = nullptr;
    OL_CHECK(olMemAlloc(DeviceHandle, OL_ALLOC_TYPE_MANAGED, Size * sizeof(T),
                        &Address));
    return ManagedBuffer<T>(Address, Size);
  }

  [[nodiscard]] std::shared_ptr<DeviceImage>
  loadBinary(const std::string &Directory, const std::string &BinaryName,
             const std::string &Extension) const;

  [[nodiscard]] std::shared_ptr<DeviceImage>
  loadBinary(const std::string &Directory, const std::string &BinaryName) const;

  template <typename KernelSignature>
  DeviceKernel<KernelSignature>
  getKernel(const std::shared_ptr<DeviceImage> &Image,
            const std::string &KernelName) const {
    if (Image->DeviceHandle != this->DeviceHandle) {
      FATAL_ERROR("Image provided to getKernel was created for a different "
                  "device");
    }

    ol_kernel_handle_t KernelHandle = nullptr;
    OL_CHECK(olGetKernel(Image->Handle, KernelName.c_str(), &KernelHandle));

    return DeviceKernel<KernelSignature>(Image, KernelHandle);
  }

  template <typename KernelSignature, typename... ArgTypes>
  void launchKernel(DeviceKernel<KernelSignature> Kernel, Dim NumGroups,
                    Dim GroupSize, ArgTypes &&...Args) const {
    using ExpectedTypes = FunctionTraits<KernelSignature>::ArgTypesTuple;
    using ProvidedTypes = std::tuple<std::decay_t<ArgTypes>...>;

    static_assert(std::is_same_v<ExpectedTypes, ProvidedTypes>,
                  "Argument types provided to launchKernel do not match the "
                  "kernel's signature");

    if (Kernel.Image->DeviceHandle != DeviceHandle) {
      FATAL_ERROR("Kernel provided to launchKernel was created for a different "
                  "device");
    }

    ol_kernel_launch_size_args_t LaunchArgs;
    LaunchArgs.Dimensions = 3; // It seems this field is not used anywhere.
                               // Defaulting to the safest value
    LaunchArgs.NumGroups = NumGroups;
    LaunchArgs.GroupSize = GroupSize;
    LaunchArgs.DynSharedMemory = 0;

    if constexpr (sizeof...(Args) == 0) {
      OL_CHECK(olLaunchKernel(nullptr, DeviceHandle, Kernel.Handle, nullptr, 0,
                              &LaunchArgs, nullptr));
    } else {
      auto KernelArgs = makeKernelArgPack(std::forward<ArgTypes>(Args)...);

      static_assert(
          (std::is_trivially_copyable_v<std::decay_t<ArgTypes>> && ...),
          "Argument types provided to launchKernel must be trivially copyable");

      OL_CHECK(olLaunchKernel(nullptr, DeviceHandle, Kernel.Handle, &KernelArgs,
                              sizeof(KernelArgs), &LaunchArgs, nullptr));
    }
  }

  [[nodiscard]] size_t getId() const { return DeviceId; }

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] std::string getPlatform() const;

private:
  [[nodiscard]] ol_platform_backend_t getBackend() const;
};

} // namespace testing
