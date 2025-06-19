#pragma once

#include "Numerics.hpp"
#include "Support.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>

namespace testing {

template <typename T> class IndexedInputRange {
  static_assert(std::is_floating_point_v<T> || std::is_integral_v<T>,
                "Type T must be an integral or floating-point type");
  static_assert(sizeof(T) <= sizeof(uint64_t),
                "Type T wider than 64 bits is unsupported");

  using StorageType = StorageTypeOf_t<T>;

private:
  StorageType MappedStart;
  StorageType MappedStop;

public:
  constexpr IndexedInputRange(T Start = getMinOrNegInf<T>(),
                              T Stop = getMaxOrInf<T>()) noexcept {
    assert((Start <= Stop) && "Start must be less than or equal to Stop");

    MappedStart = mapToOrderedUnsigned(Start);
    MappedStop = mapToOrderedUnsigned(Stop);

    assert(
        ((MappedStop - MappedStart) < std::numeric_limits<uint64_t>::max()) &&
        "The range is too large to index");
  }

  [[nodiscard]] constexpr uint64_t getSize() const noexcept {
    return static_cast<uint64_t>(MappedStop) - MappedStart + 1;
  }

  [[nodiscard]] constexpr T operator[](uint64_t Index) const noexcept {
    assert((Index < getSize()) && "Index is out of range");

    StorageType MappedValue = MappedStart + Index;
    return mapFromOrderedUnsigned(MappedValue);
  }

private:
  // Linearise T values into an ordered unsigned space:
  //  * The mapping is monotonic: a >= b if, and only if, map(a) >= map(b)
  //  * The difference |map(a) − map(b)| equals the number of representable
  //    values between a and b within the same type
  static constexpr StorageType mapToOrderedUnsigned(T Value) {
    if constexpr (std::is_floating_point_v<T>) {
      StorageType SignMask = FPUtils<T>::SignMask;
      StorageType Bits = FPUtils<T>::getAsBits(Value);
      return (Bits & SignMask) ? SignMask - (Bits - SignMask) - 1
                               : SignMask + Bits;
    }

    if constexpr (std::is_signed_v<T>) {
      StorageType SignMask = maskLeadingOnes<StorageType, 1>();
      return __builtin_bit_cast(StorageType, Value) ^ SignMask;
    }

    return Value;
  }

  static constexpr T mapFromOrderedUnsigned(StorageType MappedValue) {
    if constexpr (std::is_floating_point_v<T>) {
      StorageType SignMask = FPUtils<T>::SignMask;
      StorageType Bits = (MappedValue < SignMask)
                             ? (SignMask - MappedValue) + SignMask - 1
                             : MappedValue - SignMask;

      return FPUtils<T>::createFromBits(Bits);
    }

    if constexpr (std::is_signed_v<T>) {
      StorageType SignMask = maskLeadingOnes<StorageType, 1>();
      return __builtin_bit_cast(T, MappedValue ^ SignMask);
    }

    return MappedValue;
  }
};

template <typename... InputTypes> class InputGenerator {
public:
  virtual ~InputGenerator() noexcept = default;

  [[nodiscard]] virtual size_t
  fill(llvm::MutableArrayRef<InputTypes>... Buffers) noexcept = 0;
};

template <typename... InputTypes>
class ExhaustiveGenerator final : public InputGenerator<InputTypes...> {
  inline static constexpr size_t NumInputs = sizeof...(InputTypes);
  static_assert(NumInputs > 0, "The number of inputs must be at least 1");

  using RangeTupleType = std::tuple<IndexedInputRange<InputTypes>...>;
  using IndexArrayType = std::array<uint64_t, NumInputs>;

private:
  uint64_t Size = 1;
  RangeTupleType RangesTuple;

  IndexArrayType Strides = {};

  std::atomic<uint64_t> FlatIndexGenerator = 0;

public:
  explicit constexpr ExhaustiveGenerator(
      const IndexedInputRange<InputTypes> &...Ranges) noexcept
      : RangesTuple(Ranges...) {
    bool Overflowed = getSizeWithOverflow(Ranges..., Size);

    assert(!Overflowed && "The input space size is too large");
    assert((Size > 0) && "The input space size must be at least 1");

    IndexArrayType DimSizes = {};

    {
      size_t Index = 0;
      ((DimSizes[Index++] = Ranges.getSize()), ...);
    }

    Strides[NumInputs - 1] = 1;
    if constexpr (NumInputs > 1) {
      for (int Index = static_cast<int>(NumInputs) - 2; Index >= 0; --Index) {
        Strides[Index] = Strides[Index + 1] * DimSizes[Index + 1];
      }
    }
  }

  [[nodiscard]] size_t
  fill(llvm::MutableArrayRef<InputTypes>... Buffers) noexcept override {
    const std::array<size_t, NumInputs> BufferSizes = {Buffers.size()...};
    const size_t BufferSize = BufferSizes[0];
    assert((BufferSize != 0) && //
           std::all_of(BufferSizes.begin(), BufferSizes.end(),
                       [&](size_t S) { return S == BufferSize; }) &&
           "All buffers must have the same, non-zero size");

    uint64_t StartFlatIndex, BatchSize;
    while (true) {
      uint64_t CurrentFlatIndex =
          FlatIndexGenerator.load(std::memory_order_relaxed);
      if (CurrentFlatIndex >= Size)
        return 0;

      BatchSize = std::min<uint64_t>(BufferSize, Size - CurrentFlatIndex);
      uint64_t NextFlatIndex = CurrentFlatIndex + BatchSize;

      if (FlatIndexGenerator.compare_exchange_weak(
              CurrentFlatIndex, NextFlatIndex,
              std::memory_order_acq_rel, // Success
              std::memory_order_acquire  // Failure
              )) {
        StartFlatIndex = CurrentFlatIndex;
        break;
      }
    }

    auto BufferPtrTuple = std::make_tuple(Buffers.data()...);

#pragma omp parallel for schedule(static) num_threads(getNumThreads(BatchSize))
    for (uint64_t Offset = 0; Offset < BatchSize; ++Offset) {
      writeInputs(StartFlatIndex, Offset, BufferPtrTuple);
    }

    return static_cast<size_t>(BatchSize);
  }

private:
  static bool
  getSizeWithOverflow(const IndexedInputRange<InputTypes> &...Ranges,
                      uint64_t &Size) noexcept {
    Size = 1;
    bool Overflowed = false;

    auto Multiplier = [&](const uint64_t RangeSize) {
      if (!Overflowed) {
        Overflowed = __builtin_mul_overflow(Size, RangeSize, &Size);
      }
    };

    (Multiplier(Ranges.getSize()), ...);

    return Overflowed;
  }

  template <typename BufferPtrTupleType>
  inline void writeInputs(const uint64_t StartFlatIndex, uint64_t Offset,
                          BufferPtrTupleType &BufferPtrTuple) const noexcept {
    auto NDIndex = getNDIndex(StartFlatIndex + Offset);
    writeInputsImpl<0>(NDIndex, Offset, BufferPtrTuple);
  }

  template <size_t Index, typename BufferPtrTupleType>
  inline void
  writeInputsImpl(const IndexArrayType &NDIndex, uint64_t Offset,
                  BufferPtrTupleType &BufferPtrTuple) const noexcept {
    if constexpr (Index < NumInputs) {
      const auto &Range = std::get<Index>(RangesTuple);
      std::get<Index>(BufferPtrTuple)[Offset] = Range[NDIndex[Index]];
      writeInputsImpl<Index + 1>(NDIndex, Offset, BufferPtrTuple);
    }
  }

  constexpr IndexArrayType getNDIndex(uint64_t FlatIndex) const noexcept {
    IndexArrayType NDIndex;

    for (size_t Index = 0; Index < NumInputs; ++Index) {
      NDIndex[Index] = FlatIndex / Strides[Index];
      FlatIndex -= NDIndex[Index] * Strides[Index];
    }

    return NDIndex;
  }
};

} // namespace testing