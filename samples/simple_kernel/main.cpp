#include "OffloadUtils.hpp"
#include <iostream>
#include <vector>

#define GROUP_SIZE_X 8

const ols::Device &getCUDADevice() {
  // Select the first CUDA device
  for (const auto &CurrentDevice : ols::getDevices())
    if (CurrentDevice.IsCUDA)
      return CurrentDevice;

  std::cerr << "FATAL ERROR: In function " << __func__ << " (" << __FILE__
            << ":" << __LINE__ << "): "
            << "No CUDA devices found" << '\n';
  std::exit(EXIT_FAILURE);
}

int main() {
  const auto &CUDADevice = getCUDADevice();

  std::vector<char> DeviceBinary;
  if (!loadDeviceBinary("simple_kernel", CUDADevice, DeviceBinary)) {
    return EXIT_FAILURE;
  }

  ol_program_handle_t Program = nullptr;
  OL_CHECK(olCreateProgram(CUDADevice.Handle, DeviceBinary.data(),
                           DeviceBinary.size(), &Program));

  ol_kernel_handle_t Kernel = nullptr;
  OL_CHECK(olGetKernel(Program, "simple_kernel", &Kernel));

  ol_kernel_launch_size_args_t LaunchArgs{};
  LaunchArgs.Dimensions = 1;

  LaunchArgs.NumGroupsX = 1;
  LaunchArgs.NumGroupsY = 1;
  LaunchArgs.NumGroupsZ = 1;

  LaunchArgs.GroupSizeX = GROUP_SIZE_X;
  LaunchArgs.GroupSizeY = 1;
  LaunchArgs.GroupSizeZ = 1;

  LaunchArgs.DynSharedMemory = 0;

  void *Buffer;
  OL_CHECK(olMemAlloc(CUDADevice.Handle, OL_ALLOC_TYPE_MANAGED,
                      GROUP_SIZE_X * sizeof(int), &Buffer));

  struct {
    void *Buffer;
  } Args{Buffer};

  OL_CHECK(olLaunchKernel(nullptr, CUDADevice.Handle, Kernel, &Args,
                          sizeof(Args), &LaunchArgs, nullptr));

  int *Data = (int *)Buffer;
  for (int i = 0; i < GROUP_SIZE_X; ++i) {
    std::cout << "Data[" << i << "] = " << Data[i]
              << (Data[i] == i ? " (correct)" : " (incorrect)") << '\n';
  }

  OL_CHECK(olMemFree(Buffer));
  OL_CHECK(olDestroyProgram(Program));

  return 0;
}