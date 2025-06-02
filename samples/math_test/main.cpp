#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

template <typename UIntType, size_t Count>
inline constexpr UIntType maskTrailingOnes() {
  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  constexpr unsigned TotalBits = CHAR_BIT * sizeof(UIntType);
  static_assert(
      Count <= TotalBits,
      "Count must be less than or equal to the bit width of UIntType");

  return Count == 0 ? 0 : (UIntType(-1) >> (TotalBits - Count));
}

template <typename FloatType> struct FPLayout;

template <> struct FPLayout<float> {
  using StorageType = uint32_t;

  inline static constexpr int SignLen = 1;
  inline static constexpr int ExponentLen = 8;
  inline static constexpr int FractionLen = 23;
};

template <> struct FPLayout<double> {
  using StorageType = uint64_t;

  inline static constexpr int SignLen = 1;
  inline static constexpr int ExponentLen = 11;
  inline static constexpr int FractionLen = 52;
};

template <typename FloatType> struct FPUtils : public FPLayout<FloatType> {
  using UP = FPLayout<FloatType>;
  using StorageType = typename UP::StorageType;
  using UP::ExponentLen;
  using UP::FractionLen;
  using UP::SignLen;

  inline static constexpr StorageType SignMask =
      maskTrailingOnes<StorageType, SignLen>() << (ExponentLen + FractionLen);

  inline static constexpr FloatType createFromBits(StorageType Bits) {
    return __builtin_bit_cast(FloatType, Bits);
  }

  inline static constexpr StorageType getAsBits(FloatType Value) {
    return __builtin_bit_cast(StorageType, Value);
  }
};

template <typename FloatType>
uint64_t computeUlpDistance(FloatType X, FloatType Y) {
  static_assert(std::is_floating_point_v<FloatType>,
                "FloatType must be a floating-point type");
  using StorageType = typename FPUtils<FloatType>::StorageType;

  if (X == Y) {
    return 0;
  }

  const bool XIsNaN = std::isnan(X);
  const bool YIsNaN = std::isnan(Y);

  if (XIsNaN && YIsNaN) {
    return 0;
  }
  if (XIsNaN || YIsNaN) {
    return std::numeric_limits<uint64_t>::max();
  }

  const bool XIsInf = std::isinf(X);
  const bool YIsInf = std::isinf(Y);

  if (XIsInf && YIsInf) {
    // If execution reaches here, X != Y, so they must be opposite infinities
    return std::numeric_limits<uint64_t>::max();
  }
  if (XIsInf || YIsInf) {
    return std::numeric_limits<uint64_t>::max();
  }

  constexpr StorageType SignMask = FPUtils<FloatType>::SignMask;

  // Linearise FloatType values into an ordered unsigned space:
  //
  //  * The mapping is monotonic: a >= b if, and only if, map(a) >= map(b)
  //  * The difference |map(a) − map(b)| equals the number of std::nextafter
  //    steps between a and b within the same type
  //
  auto mapToOrderedUInt = [SignMask](FloatType Value) {
    const StorageType Bits = FPUtils<FloatType>::getAsBits(Value);
    return (Bits & SignMask) ? SignMask - (Bits - SignMask) : SignMask + Bits;
  };

  const StorageType MappedX = mapToOrderedUInt(X);
  const StorageType MappedY = mapToOrderedUInt(Y);
  return static_cast<uint64_t>(MappedX > MappedY ? MappedX - MappedY
                                                 : MappedY - MappedX);
}

int main() {
  float X = 1.0f + std::numeric_limits<float>::epsilon();
  float Y = 1.0f;

  // UlpError: 1
  std::cout << "UlpError: " << computeUlpDistance(X, Y) << '\n';

  X = std::numeric_limits<float>::denorm_min();
  Y = -X;

  // UlpError: 2
  std::cout << "UlpError: " << computeUlpDistance(X, Y) << '\n';

  return 0;
}