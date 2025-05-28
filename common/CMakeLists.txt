add_library(
  OLSCommon STATIC
  OffloadUtils.cpp
)

target_include_directories(
  OLSCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${LLVM_HOME}/include"
)

target_link_libraries(
  OLSCommon PUBLIC
  LLVMOffload dl pthread
)