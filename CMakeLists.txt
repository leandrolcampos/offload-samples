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

execute_process(
  COMMAND "${CMAKE_CXX_COMPILER}" -print-target-triple
  OUTPUT_VARIABLE LLVM_DEFAULT_TRIPLE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(LLVM_TARGET_LIBDIR "${LLVM_LIBDIR}/${LLVM_DEFAULT_TRIPLE}")
message(STATUS "LLVM target-specific library path: ${LLVM_TARGET_LIBDIR}")

link_directories("${LLVM_LIBDIR}" "${LLVM_TARGET_LIBDIR}")

set(CMAKE_INSTALL_RPATH "${LLVM_LIBDIR};${LLVM_TARGET_LIBDIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

list(APPEND CMAKE_PREFIX_PATH "${LLVM_HOME}")

# -----------------------------------------------------------------------------
# LLVM Libc integration
# -----------------------------------------------------------------------------

set(
  LLVM_LIBC_ARCHIVE
  "${LLVM_TARGET_LIBDIR}/libllvmlibc.a" CACHE FILEPATH
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

find_library(
  LLVMSupport
  NAMES LLVMSupport
  PATHS "${LLVM_HOME}/lib"
  REQUIRED
)

find_library(
  LLVMDemangle
  NAMES LLVMDemangle
  PATHS "${LLVM_HOME}/lib"
  REQUIRED
)

find_package(OpenMP REQUIRED)

# -----------------------------------------------------------------------------
# Subdirectory
# -----------------------------------------------------------------------------

add_subdirectory(common)
add_subdirectory(samples/device_query)
add_subdirectory(samples/math_test)
add_subdirectory(samples/simple_kernel)
add_subdirectory(samples/testing)