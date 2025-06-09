#include "Tester.hpp"
#include "UlpDistance.hpp"
#include <chrono>
#include <climits>
#include <iostream>

using namespace ols::testing;

#define CHECK_ULP_DISTANCE(A, B, EXPECTED)                                     \
  do {                                                                         \
    const uint64_t actual = computeUlpDistance((A), (B));                      \
    if (actual != (EXPECTED)) {                                                \
      std::cerr << "--- ULP CHECK FAILED ---\n"                                \
                << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n"      \
                << "Check: computeUlpDistance(" << #A << ", " << #B << ")\n"   \
                << "  Expected: " << (EXPECTED) << "\n"                        \
                << "  Actual:   " << actual << "\n\n";                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

static void checkUlpDistance() {
  constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float Inf = std::numeric_limits<float>::infinity();
  constexpr float MinDenorm = std::numeric_limits<float>::denorm_min();
  constexpr float MaxFinite = std::numeric_limits<float>::max();
  const float NextAfterOne = std::nextafter(1.0f, Inf);

  CHECK_ULP_DISTANCE(1.0f, 1.0f, 0U);

  CHECK_ULP_DISTANCE(+0.0f, +0.0f, 0U);
  CHECK_ULP_DISTANCE(-0.0f, -0.0f, 0U);
  CHECK_ULP_DISTANCE(-0.0f, +0.0f, 1U);

  CHECK_ULP_DISTANCE(NaN, NaN, 0U);
  CHECK_ULP_DISTANCE(NaN, 1.0f, UINT64_MAX);

  CHECK_ULP_DISTANCE(Inf, Inf, 0U);
  CHECK_ULP_DISTANCE(Inf, MaxFinite, 1U);
  CHECK_ULP_DISTANCE(-Inf, -Inf, 0U);
  CHECK_ULP_DISTANCE(-Inf, -MaxFinite, 1U);

  CHECK_ULP_DISTANCE(-Inf, Inf, 4'278'190'080U);
  CHECK_ULP_DISTANCE(-MaxFinite, MaxFinite, 4'278'190'078U);

  CHECK_ULP_DISTANCE(1.0f, NextAfterOne, 1U);
  CHECK_ULP_DISTANCE(-MinDenorm, MinDenorm, 2U);
}

int main() {
  checkUlpDistance();

  const auto StartTime = std::chrono::steady_clock::now();

  UnaryOpExhaustiveTester<logf> LogfTester;
  const auto Result = LogfTester.testPositiveRange();

  const auto EndTime = std::chrono::steady_clock::now();
  const auto Duration = EndTime - StartTime;

  const std::chrono::duration<double> ElapsedSeconds = Duration;

  std::cout << "MaxUlpDistance: " << Result.MaxUlpDistance << '\n';
  std::cout << "FailureCount..: " << Result.FailureCount << '\n';

  if (Result.WorstInput.has_value()) {
    const float WorstInput = Result.WorstInput.value();
    std::cout << "WorstInput....: " << WorstInput << " (" << std::hexfloat
              << WorstInput << ")\n";
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Execution time: " << ElapsedSeconds << '\n';

  return 0;
}