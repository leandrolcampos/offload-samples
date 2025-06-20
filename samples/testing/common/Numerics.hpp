#pragma once

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace testing {
namespace internal {

template <typename T> struct TypeIdentityOf {
  using type = T;
};

} // namespace internal

template <typename T> struct StorageTypeOf {
private:
  static constexpr auto getStorageType() noexcept {
    if constexpr (std::is_same_v<T, float>) {
      return internal::TypeIdentityOf<uint32_t>{};

    } else if constexpr (std::is_same_v<T, double>) {
      return internal::TypeIdentityOf<uint64_t>{};

    } else if constexpr (std::is_unsigned_v<T>) {
      return internal::TypeIdentityOf<T>{};

    } else if constexpr (std::is_signed_v<T>) {
      return internal::TypeIdentityOf<std::make_unsigned_t<T>>{};

    } else {
      static_assert(!std::is_same_v<T, T>, "Unsupported type");
    }
  }

public:
  using type = typename decltype(getStorageType())::type;
};

template <typename T> using StorageTypeOf_t = typename StorageTypeOf<T>::type;

template <typename T> constexpr T getMinOrNegInf() noexcept {
  static_assert(std::is_arithmetic_v<T>, "Type T must be an arithmetic type");

  if constexpr (std::is_floating_point_v<T> &&
                std::numeric_limits<T>::has_infinity) {
    return -std::numeric_limits<T>::infinity();
  }

  return std::numeric_limits<T>::lowest();
}

template <typename T> constexpr T getMaxOrInf() noexcept {
  static_assert(std::is_arithmetic_v<T>, "Type T must be an arithmetic type");

  if constexpr (std::is_floating_point_v<T> &&
                std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  }

  return std::numeric_limits<T>::max();
}

template <typename UIntType, size_t Count>
inline constexpr UIntType maskLeadingOnes() {
  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  constexpr unsigned TotalBits = CHAR_BIT * sizeof(UIntType);
  static_assert(
      Count <= TotalBits,
      "Count must be less than or equal to the bit width of UIntType");

  return Count == 0 ? UIntType(0) : (~UIntType(0) << (TotalBits - Count));
  ;
}

template <typename UIntType, size_t Count>
inline constexpr UIntType maskTrailingOnes() {
  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  constexpr unsigned TotalBits = CHAR_BIT * sizeof(UIntType);
  static_assert(
      Count <= TotalBits,
      "Count must be less than or equal to the bit width of UIntType");

  return Count == 0 ? UIntType(0) : (~UIntType(0) >> (TotalBits - Count));
}

template <typename FloatType> struct FPLayout;

template <> struct FPLayout<float> {
  inline static constexpr size_t SignLen = 1;
  inline static constexpr size_t ExponentLen = 8;
  inline static constexpr size_t FractionLen = 23;
};

template <> struct FPLayout<double> {
  inline static constexpr size_t SignLen = 1;
  inline static constexpr size_t ExponentLen = 11;
  inline static constexpr size_t FractionLen = 52;
};

template <typename FloatType> struct FPUtils : public FPLayout<FloatType> {
  using Layout = FPLayout<FloatType>;
  using StorageType = StorageTypeOf_t<FloatType>;
  using Layout::ExponentLen;
  using Layout::FractionLen;
  using Layout::SignLen;

  inline static constexpr StorageType SignMask =
      maskTrailingOnes<StorageType, SignLen>() << (ExponentLen + FractionLen);

  static constexpr FloatType createFromBits(StorageType Bits) noexcept {
    return __builtin_bit_cast(FloatType, Bits);
  }

  static constexpr StorageType getAsBits(FloatType Value) noexcept {
    return __builtin_bit_cast(StorageType, Value);
  }
};

template <typename FloatType>
uint64_t computeUlpDistance(FloatType X, FloatType Y) {
  static_assert(std::is_floating_point_v<FloatType>,
                "FloatType must be a floating-point type");
  using StorageType = typename FPUtils<FloatType>::StorageType;

  if (X == Y) {
    if (std::signbit(X) != std::signbit(Y)) [[unlikely]] {
      // When X == Y, different sign bits imply that X and Y are +0.0 and -0.0
      // (in any order). Since we want to treat them as unequal in the context
      // of accuracy testing of mathematical functions, we return the smallest
      // non-zero value
      return 1U;
    }
    return 0U;
  }

  const bool XIsNaN = std::isnan(X);
  const bool YIsNaN = std::isnan(Y);

  if (XIsNaN && YIsNaN) {
    return 0U;
  }
  if (XIsNaN || YIsNaN) {
    return std::numeric_limits<uint64_t>::max();
  }

  constexpr StorageType SignMask = FPUtils<FloatType>::SignMask;

  // Linearise FloatType values into an ordered unsigned space:
  //  * The mapping is monotonic: a >= b if, and only if, map(a) >= map(b)
  //  * The difference |map(a) − map(b)| equals the number of std::nextafter
  //    steps between a and b within the same type
  auto MapToOrderedUnsigned = [SignMask](FloatType Value) {
    const StorageType Bits = FPUtils<FloatType>::getAsBits(Value);
    return (Bits & SignMask) ? SignMask - (Bits - SignMask) : SignMask + Bits;
  };

  const StorageType MappedX = MapToOrderedUnsigned(X);
  const StorageType MappedY = MapToOrderedUnsigned(Y);
  return static_cast<uint64_t>(MappedX > MappedY ? MappedX - MappedY
                                                 : MappedY - MappedX);
}

} // namespace testing