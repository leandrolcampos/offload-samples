#pragma once

#include <llvm/Support/raw_ostream.h>
#include <string>

#define FATAL_ERROR(Message)                                                   \
  testing::internal::fatalErrorHandler(Message, __FILE__, __LINE__, __func__)

namespace testing {
namespace internal {

[[noreturn]] inline void fatalErrorHandler(const std::string &Message,
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

template <typename ReturnType, typename... ArgTypes>
struct FunctionTraits<ReturnType(ArgTypes...)> {
  using RetType = ReturnType;
  using ArgTuple = std::tuple<ArgTypes...>;

  static constexpr size_t ArgCount = sizeof...(ArgTypes);
};

template <typename ReturnType, typename... ArgTypes>
struct FunctionTraits<ReturnType (*)(ArgTypes...)>
    : FunctionTraits<ReturnType(ArgTypes...)> {};

} // namespace testing