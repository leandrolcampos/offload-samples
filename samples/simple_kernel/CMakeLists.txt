add_subdirectory(device_code)

add_executable(simple-kernel main.cpp)
add_dependencies(simple-kernel SimpleKernelDeviceBins)

target_link_libraries(simple-kernel PRIVATE OLSCommon)