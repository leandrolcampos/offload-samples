#pragma once

#include "OffloadUtils.hpp"
#include "UlpDistance.hpp"
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace ols {
namespace testing {

template <auto Func> struct OperationInfo;

template <auto Func> struct OperationTraits;

template <typename RetType, typename ArgType, RetType (*Func)(ArgType)>
struct OperationTraits<Func> {
  using OutType = RetType;
  using InType = ArgType;

  inline static constexpr size_t InputCount = 1;
};

template <auto Func>
struct UnaryOp : public OperationInfo<Func>, public OperationTraits<Func> {
  using Info = OperationInfo<Func>;
  using Traits = OperationTraits<Func>;

  using OutType = typename Traits::OutType;
  using InType = typename Traits::InType;

  using Info::BinaryName;
  using Info::KernelName;
  using Info::UlpTolerance;
  using Traits::InputCount;

  static_assert(InputCount == 1, "Func must be a unary operation");
};

template <typename... BufferTypes>
static constexpr size_t getDefaultBatchSize() {
  static_assert(sizeof...(BufferTypes) > 0,
                "At least one buffer type must be provided");

  constexpr size_t TotalMemory = 1ULL * 1024 * 1024 * 1024; // 1 GB
  constexpr size_t ElementTupleSize = (sizeof(BufferTypes) + ...);
  return TotalMemory / ElementTupleSize;
}

template <typename T> struct CheckResult {
  std::optional<T> WorstInput;
  uint64_t MaxUlpDistance = 0U;
  uint64_t FailureCount = 0U;
};

template <auto Func> class UnaryOpChecker : public UnaryOp<Func> {
  using Op = UnaryOp<Func>;

public:
  using OutType = typename Op::OutType;
  using InType = typename Op::InType;

  using Op::BinaryName;
  using Op::InputCount;
  using Op::KernelName;
  using Op::UlpTolerance;

  UnaryOpChecker(size_t BufferSize)
      : GPUDevice(getCUDADevice()), BufferSize(BufferSize) {
    Host = getHostHandle();

    OL_CHECK(olMemAlloc(GPUDevice.Handle, OL_ALLOC_TYPE_DEVICE,
                        BufferSize * sizeof(InType), &InBuffer));
    OL_CHECK(olMemAlloc(GPUDevice.Handle, OL_ALLOC_TYPE_MANAGED,
                        BufferSize * sizeof(OutType), &OutBuffer));

    std::vector<char> DeviceBinary;
    loadDeviceBinary(BinaryName, GPUDevice, DeviceBinary);
    OL_CHECK(olCreateProgram(GPUDevice.Handle, DeviceBinary.data(),
                             DeviceBinary.size(), &Program));
    OL_CHECK(olGetKernel(Program, KernelName, &Kernel));
  }

  ~UnaryOpChecker() {
    OL_CHECK(olMemFree(InBuffer));
    OL_CHECK(olMemFree(OutBuffer));
    OL_CHECK(olDestroyProgram(Program));
  }

  size_t getBufferSize() const { return BufferSize; }

  void check(const std::vector<InType> &Input, CheckResult<InType> &Result) {
    size_t InputSize = Input.size();
    if (InputSize > BufferSize) {
      FATAL_ERROR(
          "Input size exceeds buffer size: " + std::to_string(InputSize) +
          " > " + std::to_string(BufferSize));
    }

    OL_CHECK(olMemcpy(nullptr, InBuffer, GPUDevice.Handle,
                      const_cast<InType *>(Input.data()), Host,
                      InputSize * sizeof(InType), nullptr));

    ol_kernel_launch_size_args_t LaunchArgs = getKernelLaunchArgs(InputSize);

    struct {
      void *In;
      void *Out;
      size_t NumElements;
    } Args{InBuffer, OutBuffer, InputSize};

    OL_CHECK(olLaunchKernel(nullptr, GPUDevice.Handle, Kernel, &Args,
                            sizeof(Args), &LaunchArgs, nullptr));

    OutType *OutData = (OutType *)OutBuffer;
    for (size_t Index = 0; Index < InputSize; Index++) {
      OutType Actual = OutData[Index];
      OutType Expected = Func(Input[Index]);
      uint64_t UlpDistance = computeUlpDistance(Actual, Expected);

      if (UlpDistance > Result.MaxUlpDistance) {
        Result.MaxUlpDistance = UlpDistance;
        Result.WorstInput = Input[Index];
      }
      if (UlpDistance > UlpTolerance) {
        Result.FailureCount++;
      }
    }
  }

private:
  ol_device_handle_t Host = nullptr;
  const Device &GPUDevice;
  ol_program_handle_t Program = nullptr;
  ol_kernel_handle_t Kernel = nullptr;
  const size_t BufferSize;
  void *InBuffer = nullptr;
  void *OutBuffer = nullptr;

  ol_kernel_launch_size_args_t getKernelLaunchArgs(uint32_t InputSize) {
    ol_kernel_launch_size_args_t LaunchArgs;
    LaunchArgs.Dimensions = 1;

    LaunchArgs.GroupSize = {1024, 1, 1};
    LaunchArgs.NumGroups = {(InputSize + LaunchArgs.GroupSize.x - 1) /
                                LaunchArgs.GroupSize.x,
                            1, 1};
    LaunchArgs.DynSharedMemory = 0;

    return LaunchArgs;
  }
};

template <auto Func>
class UnaryOpExhaustiveTester : public UnaryOpChecker<Func> {
  using Checker = UnaryOpChecker<Func>;
  using OutType = typename Checker::OutType;
  using InType = typename Checker::InType;
  using StorageType = typename FPUtils<InType>::StorageType;

  using Checker::BinaryName;
  using Checker::InputCount;
  using Checker::KernelName;
  using Checker::UlpTolerance;

  inline static constexpr size_t DefaultBufferSize =
      getDefaultBatchSize<OutType, InType>();

public:
  UnaryOpExhaustiveTester(size_t BufferSize = DefaultBufferSize)
      : UnaryOpChecker<Func>(BufferSize) {};

  template <StorageType Start, StorageType End>
  CheckResult<InType> testCustomRange() {
    static_assert(Start <= End, "Start must be less than or equal to End");
    static_assert(End < std::numeric_limits<StorageType>::max(),
                  "End must be less than the maximum value of StorageType");

    std::vector<InType> Input;
    auto BufferSize = this->getBufferSize();
    Input.reserve(BufferSize);
    CheckResult<InType> Result{};

    for (StorageType XBits = Start; XBits <= End; XBits++) {
      Input.emplace_back(FPUtils<InType>::createFromBits(XBits));

      if (Input.size() == BufferSize) {
        this->check(Input, Result);
        Input.clear();
      }
    }

    if (!Input.empty()) {
      this->check(Input, Result);
    }

    return Result;
  }

  CheckResult<InType> testPositiveRange() {
    StorageType constexpr Start = 0U;
    StorageType constexpr End =
        FPUtils<InType>::getAsBits(std::numeric_limits<InType>::infinity());

    return testCustomRange<Start, End>();
  }
};

template <> struct OperationInfo<logf> {
  inline static constexpr const char *BinaryName = "MathTest";
  inline static constexpr const char *KernelName = "applyLogf";
  inline static constexpr uint64_t UlpTolerance = 3U;
};
} // namespace testing
} // namespace ols