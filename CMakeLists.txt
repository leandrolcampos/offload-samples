cmake_minimum_required(VERSION 3.20)
project(offload-samples LANGUAGES CXX)

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(-fno-builtin)

# -----------------------------------------------------------------------------
# LLVM installation paths
# -----------------------------------------------------------------------------

set(LLVM_HOME "$ENV{LLVM_HOME}" CACHE PATH "Path to the LLVM installation")
set(LLVM_LIBDIR "${LLVM_HOME}/lib")

# Make LLVM libraries discoverable at build and runtime
list(APPEND CMAKE_PREFIX_PATH "${LLVM_HOME}")
set(CMAKE_BUILD_RPATH "${LLVM_LIBDIR}")

# -----------------------------------------------------------------------------
# LLVM Libc integration
# -----------------------------------------------------------------------------

execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -print-target-triple
  OUTPUT_VARIABLE LLVM_DEFAULT_TRIPLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE LLVM_TRIPLE_RESULT
  ERROR_QUIET
)

set(
  LLVM_LIBC_ARCHIVE
  "${LLVM_LIBDIR}/${LLVM_DEFAULT_TRIPLE}/libllvmlibc.a" CACHE FILEPATH
  "Full path to libllvmlibc.a"
)

add_library(LLVMLibC STATIC IMPORTED GLOBAL)
set_target_properties(
  LLVMLibC
  PROPERTIES IMPORTED_LOCATION "${LLVM_LIBC_ARCHIVE}"
)

# ---------------------------------------------------------------------------
# Device code staging directory
# ---------------------------------------------------------------------------

set(OLS_DEVICE_CODE_PATH "${CMAKE_BINARY_DIR}/device_code")
file(MAKE_DIRECTORY "${OLS_DEVICE_CODE_PATH}")
set_property(GLOBAL PROPERTY OLS_DEVICE_CODE_PATH "${OLS_DEVICE_CODE_PATH}")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

find_library(
  LLVMOffload
  NAMES LLVMOffload
  PATHS "${LLVM_HOME}/lib"
  REQUIRED
)

# -----------------------------------------------------------------------------
# Subdirectory
# -----------------------------------------------------------------------------

add_subdirectory(common)
add_subdirectory(samples/device_query)
add_subdirectory(samples/math_test)
add_subdirectory(samples/simple_kernel)