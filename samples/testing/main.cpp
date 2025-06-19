#include "DeviceContext.hpp"
#include "InputGenerator.hpp"
#include <iostream>

using namespace testing;

#define ASSERT_EQUAL(A, B)                                                     \
  do {                                                                         \
    const auto ValA = (A);                                                     \
    const auto ValB = (B);                                                     \
    if (ValA != ValB) {                                                        \
      llvm::errs() << "--- ASSERT_EQUAL FAILED ---\n"                          \
                   << "File: " << __FILE__ << ", Line: " << __LINE__ << "\n"   \
                   << "Check: " << #A << " == " << #B << "\n"                  \
                   << "  LHS: " << ValA << "\n"                                \
                   << "  RHS: " << ValB << "\n\n";                             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (false)

namespace {

void devices() {
  std::cout << "--- DEVICE ---\n\n"
            << "Number of devices: " << countDevices() << "\n";

  if (countDevices() > 0) {
    DeviceContext Context;
    std::cout << "  DeviceId: 0\n"
              << "  Name:     " << Context.getName() << "\n"
              << "  Platform: " << Context.getPlatform() << "\n\n";
  }
}

void helloWorld() {
  std::cout << "--- HELLO WORLD ---\n\n";

  const std::string DeviceBinsDirectory = DEVICE_CODE_PATH;
  DeviceContext Context;
  auto Image = Context.loadBinary(DeviceBinsDirectory, "HelloWorld");
  auto Kernel = Context.getKernel<void()>(Image, "printHelloWorld");
  Context.launchKernel(Kernel, 4, 2);

  std::cout << "\n\n";
}

void testIndexedInputRangeForFloat() {
  IndexedInputRange<float> Range;
  ASSERT_EQUAL(Range.getSize(), 4278190082);
  ASSERT_EQUAL(Range[0], -std::numeric_limits<float>::infinity());
  ASSERT_EQUAL(__builtin_bit_cast(uint32_t, Range[2139095040]),
               __builtin_bit_cast(uint32_t, -0.0f));
  ASSERT_EQUAL(__builtin_bit_cast(uint32_t, Range[2139095041]),
               __builtin_bit_cast(uint32_t, +0.0f));
  ASSERT_EQUAL(Range[Range.getSize() - 1],
               std::numeric_limits<float>::infinity());
}

void testIndexedInputRangeForDouble() {
  IndexedInputRange<double> Range;
  ASSERT_EQUAL(Range.getSize(), 18437736874454810626ULL);
  ASSERT_EQUAL(Range[0], -std::numeric_limits<double>::infinity());
  ASSERT_EQUAL(__builtin_bit_cast(uint64_t, Range[9218868437227405312ULL]),
               __builtin_bit_cast(uint64_t, -0.0));
  ASSERT_EQUAL(__builtin_bit_cast(uint64_t, Range[9218868437227405313ULL]),
               __builtin_bit_cast(uint64_t, +0.0));
  ASSERT_EQUAL(Range[Range.getSize() - 1],
               std::numeric_limits<double>::infinity());
}

void testIndexedInputRangeForInt32() {
  IndexedInputRange<int32_t> Range;

  ASSERT_EQUAL(Range.getSize(), 4294967296);
  ASSERT_EQUAL(Range[0], std::numeric_limits<int32_t>::lowest());
  ASSERT_EQUAL(Range[2147483648], 0);
  ASSERT_EQUAL(Range[Range.getSize() - 1], std::numeric_limits<int32_t>::max());
}

void testExhaustiveGeneratorUnequalBatches() {
  constexpr size_t BufferSize = 5;

  int32_t Start = -10;
  int32_t Stop = Start + 4 * BufferSize; // 10 => Size = 21

  IndexedInputRange<int32_t> Range(Start, Stop);
  ExhaustiveGenerator Generator(Range);

  std::array<int32_t, BufferSize> Buffer;
  auto BufferRef = llvm::MutableArrayRef<int32_t>(Buffer.data(), Buffer.size());

  auto Remainder = static_cast<size_t>(Stop - Start + 1);
  auto NumBatches = (Remainder + BufferSize - 1) / BufferSize;
  size_t BatchCount = 0;

  while (size_t BatchSize = Generator.fill(BufferRef)) {
    ASSERT_EQUAL(BatchSize, std::min(BufferSize, Remainder));
    Remainder -= BatchSize;
    BatchCount++;
  }

  ASSERT_EQUAL(Remainder, 0);
  ASSERT_EQUAL(BatchCount, NumBatches);
}

void testExhaustiveGeneratorEqualBatches() {
  constexpr size_t BufferSize = 5;

  constexpr int32_t Start = -10;
  constexpr int32_t Stop = Start + 4 * BufferSize - 1; // 9 => Size = 20

  constexpr IndexedInputRange<int32_t> Range(Start, Stop);
  ExhaustiveGenerator Generator(Range);

  std::array<int32_t, BufferSize> Buffer;
  auto BufferRef = llvm::MutableArrayRef<int32_t>(Buffer.data(), Buffer.size());

  auto Remainder = static_cast<size_t>(Stop - Start + 1);
  auto NumBatches = (Remainder + BufferSize - 1) / BufferSize;
  size_t BatchCount = 0;

  while (size_t BatchSize = Generator.fill(BufferRef)) {
    ASSERT_EQUAL(BatchSize, BufferSize);
    Remainder -= BatchSize;
    BatchCount++;
  }

  ASSERT_EQUAL(Remainder, 0);
  ASSERT_EQUAL(BatchCount, NumBatches);
}

void testExhaustiveGenerator1D() {
  constexpr float MinDenorm = std::numeric_limits<float>::denorm_min();
  constexpr float Inf = std::numeric_limits<float>::infinity();
  const float Start = std::nextafter(-MinDenorm, -Inf);
  const float Stop = std::nextafter(MinDenorm, +Inf);

  const IndexedInputRange Range(Start, Stop);
  ExhaustiveGenerator Generator(Range);

  ASSERT_EQUAL(Range.getSize(), 6);

  constexpr size_t BufferSize = 6;
  std::array<float, BufferSize> Buffer;
  auto BufferRef = llvm::MutableArrayRef<float>(Buffer.data(), Buffer.size());

  auto BatchSize = Generator.fill(BufferRef);

  ASSERT_EQUAL(BatchSize, BufferSize);

  for (size_t Index = 0; Index < BatchSize; Index++) {
    ASSERT_EQUAL(Buffer[Index], Range[Index]);
  }

  ASSERT_EQUAL(Generator.fill(BufferRef), 0);
}

void testExhaustiveGenerator2D() {
  constexpr float MinDenorm = std::numeric_limits<float>::denorm_min();
  constexpr float Inf = std::numeric_limits<float>::infinity();
  const float Start = std::nextafter(-MinDenorm, -Inf);
  const float Stop = std::nextafter(MinDenorm, +Inf);

  const IndexedInputRange RangeX(Start, Stop);
  ASSERT_EQUAL(RangeX.getSize(), 6);

  const auto RangeY = RangeX;
  ExhaustiveGenerator Generator(RangeX, RangeY);

  constexpr size_t BufferSize = 6 * 6;
  std::array<float, BufferSize> BufferX, BufferY;
  auto BufferXRef =
      llvm::MutableArrayRef<float>(BufferX.data(), BufferX.size());
  auto BufferYRef =
      llvm::MutableArrayRef<float>(BufferY.data(), BufferY.size());

  auto BatchSize = Generator.fill(BufferXRef, BufferYRef);

  ASSERT_EQUAL(BatchSize, BufferSize);

  size_t GlobalIndex = 0;
  for (size_t IndexX = 0; IndexX < RangeX.getSize(); IndexX++) {
    for (size_t IndexY = 0; IndexY < RangeY.getSize(); IndexY++) {
      ASSERT_EQUAL(BufferXRef[GlobalIndex], RangeX[IndexX]);
      ASSERT_EQUAL(BufferYRef[GlobalIndex], RangeY[IndexY]);
      GlobalIndex++;
    }
  }

  ASSERT_EQUAL(Generator.fill(BufferXRef, BufferYRef), 0);
}

} // namespace

int main() {
  devices();
  helloWorld();
  testIndexedInputRangeForFloat();
  testIndexedInputRangeForDouble();
  testIndexedInputRangeForInt32();
  testExhaustiveGeneratorUnequalBatches();
  testExhaustiveGeneratorEqualBatches();
  testExhaustiveGenerator1D();
  testExhaustiveGenerator2D();

  return 0;
}