#pragma once

#include "Support.hpp"
#include <cstdint>
#include <omp.h>
#include <optional>
#include <tuple>

namespace testing {

template <typename OutType, typename... InTypes> struct TestResult {
  uint64_t TestCaseCount = 0;

  uint64_t FailureCount = 0;
  uint64_t MaxUlpDistance = 0;

  struct TestCase {
    OutType Expected;
    OutType Actual;
    std::tuple<InTypes...> Inputs;
  };

  std::optional<TestCase> WorstTestCase;

  void aggregate(const TestResult &Other) noexcept {
    TestCaseCount += Other.TestCaseCount;
    FailureCount += Other.FailureCount;

    if (Other.MaxUlpDistance > MaxUlpDistance) {
      MaxUlpDistance = Other.MaxUlpDistance;
      WorstTestCase = Other.WorstTestCase;
    }
  }
};

template <auto Func> class MathChecker {
  using Traits = FunctionTraits<decltype(Func)>;
  using OutType = typename Traits::ReturnType;
  using InTypesTuple = typename Traits::ArgTypesTuple;

  using Config = FunctionConfig<Func>;
  using Config::UlpTolerance;

  template <typename... Ts>
  using BufferTupleType = std::tuple<llvm::ArrayRef<Ts>...>;
  using InBuffersTupleType = ApplyTupleTypes_t<InTypesTuple, BufferTupleType>;

  template <typename... Ts>
  using PartialResultType = TestResult<OutType, Ts...>;
  using ResultType = ApplyTupleTypes_t<InTypesTuple, PartialResultType>;

public:
  MathChecker() = delete;

  [[nodiscard]] static ResultType
  check(const InBuffersTupleType &InBuffers,
        const llvm::ArrayRef<OutType> &OutBuffer) noexcept {
    const size_t BufferSize = OutBuffer.size();
    std::apply(
        [&](const auto &...Buffers) {
          assert(
              ((Buffers.size() == BufferSize) && ...) &&
              "All input buffers must have the same size as the output buffer");
        },
        InBuffers);

    assert((BufferSize != 0) && "Buffer size cannot be zero");

    ResultType AggregatedResult;
    const size_t NumThreads = getNumThreads(BufferSize);

#pragma omp parallel num_threads(NumThreads)
    {
      ResultType ThreadResult;

#pragma omp for schedule(static)
      for (size_t Index = 0; Index < BufferSize; ++Index) {
        auto CurrentInputsTuple = std::apply(
            [&](const auto &...Buffers) {
              return std::make_tuple(Buffers[Index]...);
            },
            InBuffers);

        OutType ExpectedOutput = std::apply(Func, CurrentInputsTuple);
        OutType ActualOutput = OutBuffer[Index];

        uint64_t UlpDistance = computeUlpDistance(ActualOutput, ExpectedOutput);

        if (UlpDistance > ThreadResult.MaxUlpDistance) {
          ThreadResult.MaxUlpDistance = UlpDistance;
          ThreadResult.WorstTestCase =
              typename ResultType::TestCase{.Expected = ExpectedOutput,
                                            .Actual = ActualOutput,
                                            .Inputs = CurrentInputsTuple};
        }

        if (UlpDistance > UlpTolerance) {
          ThreadResult.FailureCount++;
        }

        ThreadResult.TestCaseCount++;
      }

#pragma omp critical
      {
        AggregatedResult.aggregate(ThreadResult);
      }
    }

    return AggregatedResult;
  }
};

} // namespace testing