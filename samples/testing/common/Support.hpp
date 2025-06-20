#pragma once

#include "llvm/Support/raw_ostream.h"
#include <omp.h>
#include <string>

#define FATAL_ERROR(Message)                                                   \
  testing::internal::reportFatalError(Message, __FILE__, __LINE__, __func__)

namespace testing {
namespace internal {

[[noreturn]] inline void reportFatalError(const std::string &Message,
                                          const char *File, int Line,
                                          const char *FuncName) {
  llvm::errs() << "--- FATAL ERROR ---\n"
               << "Location: " << File << ":" << Line << "\n"
               << "Function: " << FuncName << "\n"
               << "Message: " << Message << "\n\n";
  std::exit(EXIT_FAILURE);
}

} // namespace internal

template <typename T> struct FunctionTraits;

template <typename RetType, typename... ArgTypes>
struct FunctionTraits<RetType(ArgTypes...)> {
  using ReturnType = RetType;
  using ArgTypesTuple = std::tuple<ArgTypes...>;

  static constexpr size_t NumArgs = sizeof...(ArgTypes);
};

template <typename RetType, typename... ArgTypes>
struct FunctionTraits<RetType (*)(ArgTypes...)>
    : FunctionTraits<RetType(ArgTypes...)> {};

template <auto Func> struct FunctionConfig;

template <typename... ArgTypes> struct KernelArgPack;

template <typename ArgType> struct KernelArgPack<ArgType> {
  std::decay_t<ArgType> Arg0;
};

template <typename ArgType0, typename ArgType1, typename... ArgTypes>
struct KernelArgPack<ArgType0, ArgType1, ArgTypes...> {
  std::decay_t<ArgType0> Arg0;
  KernelArgPack<ArgType1, ArgTypes...> Args;
};

template <typename... ArgTypes>
KernelArgPack<ArgTypes...> makeKernelArgPack(ArgTypes &&...Args) {
  return {std::forward<ArgTypes>(Args)...};
}

template <typename TupleTypes, template <typename...> class Template>
struct ApplyTupleTypes;

template <template <typename...> class Template, typename... Ts>
struct ApplyTupleTypes<std::tuple<Ts...>, Template> {
  using type = Template<Ts...>;
};

template <typename TupleTypes, template <typename...> class Template>
using ApplyTupleTypes_t = typename ApplyTupleTypes<TupleTypes, Template>::type;

template <typename UIntType>
static size_t getNumThreads(UIntType ProblemSize,
                            UIntType GrainSize = 32768) noexcept {
  // GrainSize determines the minimum number of elements to warrant an
  // additional thread. Its default value was copied from:
  // https://github.com/pytorch/pytorch/blob/20dfce591ce88bc957ffcd0c8dc7d5f7611a4a3b/aten/src/ATen/TensorIterator.h#L86

  static_assert(std::is_unsigned_v<UIntType>,
                "UIntType must be an unsigned integer type");

  const size_t MaxThreads = omp_get_max_threads();
  const size_t DesiredThreads =
      static_cast<size_t>((ProblemSize + GrainSize - 1) / GrainSize);

  // Ensure at least one thread is always returned
  return std::max<size_t>(1, std::min(MaxThreads, DesiredThreads));
}

} // namespace testing