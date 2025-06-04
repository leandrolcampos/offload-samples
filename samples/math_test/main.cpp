#include "UlpDistance.hpp"
#include <climits>
#include <cmath>
#include <iostream>

using namespace ols::testing;

int main() {
  constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float Inf = std::numeric_limits<float>::infinity();
  constexpr float MinDenorm = std::numeric_limits<float>::denorm_min();
  constexpr float MaxFinite = std::numeric_limits<float>::max();
  const float NextAfterOne = std::nextafter(1.0f, Inf);

  // UlpError: 0
  std::cout << "ulp(1.0, 1.0) = " << computeUlpDistance(1.0f, 1.0f) << '\n';

  // UlpError: 0
  std::cout << "ulp(+0.0, +0.0) = " << computeUlpDistance(+0.0f, +0.0f) << '\n';

  // UlpError: 0
  std::cout << "ulp(-0.0, -0.0) = " << computeUlpDistance(-0.0f, -0.0f) << '\n';

  // UlpError: 1
  std::cout << "ulp(-0.0, +0.0) = " << computeUlpDistance(-0.0f, +0.0f) << '\n';

  // UlpError: 0
  std::cout << "ulp(NaN, NaN) = " << computeUlpDistance(NaN, NaN) << '\n';

  // UlpError: UINT64_MAX
  std::cout << "ulp(NaN, 1.0) = " << computeUlpDistance(NaN, 1.0f) << '\n';

  // UlpError: 1
  std::cout << "ulp(Inf, MaxFinite) = " << computeUlpDistance(Inf, MaxFinite)
            << '\n';

  // UlpError: 1
  std::cout << "ulp(-Inf, -MaxFinite) = "
            << computeUlpDistance(-Inf, -MaxFinite) << '\n';

  // UlpError: (Huge, but < UINT64_MAX)
  std::cout << "ulp(-MaxFinite, MaxFinite) = "
            << computeUlpDistance(-MaxFinite, MaxFinite) << '\n';

  // UlpError: (Huge, but < UINT64_MAX)
  std::cout << "ulp(-Inf, Inf) = " << computeUlpDistance(-Inf, Inf) << '\n';

  // UlpError: 1
  std::cout << "ulp(1.0, NextAfterOne) = "
            << computeUlpDistance(1.0f, NextAfterOne) << '\n';

  // UlpError: 2
  std::cout << "ulp(-MinDenorm, MinDenorm) = "
            << computeUlpDistance(-MinDenorm, MinDenorm) << '\n';

  return 0;
}