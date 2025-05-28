macro(add_device_code src_filename binary_name)
  set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${src_filename})

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    get_filename_component(CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
    set(BIN_PATH "${OLS_DEVICE_CODE_PATH}/${binary_name}.nvptx64.bin")
    add_custom_command(OUTPUT ${BIN_PATH}
      COMMAND ${CMAKE_C_COMPILER}
              --target=nvptx64-nvidia-cuda
              -march=native
              --cuda-path=${CUDA_ROOT}
              ${SRC_PATH} -o ${BIN_PATH}
      DEPENDS ${SRC_PATH})
    list(APPEND BINARIES ${BIN_PATH})
  endif()

  # TODO: Build for AMDGPU and host CPU
endmacro()

add_device_code(simple_kernel.c simple_kernel)
add_custom_target(SimpleKernelDeviceBins DEPENDS ${BINARIES})

add_executable(simple-kernel main.cpp)
add_dependencies(simple-kernel SimpleKernelDeviceBins)

target_link_libraries(simple-kernel PRIVATE OLSCommon)