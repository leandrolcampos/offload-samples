add_subdirectory(common)
add_subdirectory(device_code)

add_executable(
  testing
  main.cpp
)

add_dependencies(testing TestingDeviceBins)

target_link_libraries(
  testing PRIVATE
  MathTest
  LLVMLibC
  OpenMP::OpenMP_CXX
)

get_property(device_code_path GLOBAL PROPERTY OLS_DEVICE_CODE_PATH)
target_compile_definitions(
  testing PRIVATE
  DEVICE_CODE_PATH="${device_code_path}"
)