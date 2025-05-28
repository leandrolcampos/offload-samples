cmake_minimum_required(VERSION 3.20)
project(offload-samples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LLVM_HOME "$ENV{LLVM_HOME}" CACHE PATH
    "Path to the LLVM installation")

set(LLVM_LIBDIR "${LLVM_HOME}/lib")
set(CMAKE_BUILD_RPATH "${LLVM_LIBDIR}")

list(APPEND CMAKE_PREFIX_PATH "${LLVM_HOME}")

find_library(
  LLVMOffload
  NAMES LLVMOffload
  PATHS "${LLVM_HOME}/lib"
  REQUIRED
)

add_subdirectory(common)
add_subdirectory(samples/device_query)