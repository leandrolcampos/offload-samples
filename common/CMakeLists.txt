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

get_property(device_code_path GLOBAL PROPERTY OLS_DEVICE_CODE_PATH)
target_compile_definitions(
  OLSCommon PRIVATE
  DEVICE_CODE_PATH="${device_code_path}"
)