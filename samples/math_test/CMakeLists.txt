add_subdirectory(device_code)

add_executable(
  math-test
  main.cpp
)

add_dependencies(math-test MathTestDeviceBins)

target_link_libraries(
  math-test PRIVATE
  OLSCommon
  LLVMLibC
)