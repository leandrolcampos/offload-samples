add_library(
  MathTest STATIC
  DeviceContext.cpp
)

target_include_directories(
  MathTest PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${LLVM_HOME}/include"
)

target_link_libraries(
  MathTest PUBLIC
  LLVMOffload dl pthread
  LLVMSupport
  LLVMDemangle
)