#include "OffloadUtils.hpp"
#include <iostream>
#include <vector>

#define GROUP_SIZE_X 8

static const ols::Device &getCUDADevice() {
  // Select the first CUDA device
  for (const auto &CurrentDevice : ols::getDevices())
    if (CurrentDevice.IsCUDA)
      return CurrentDevice;

  FATAL_ERROR("No CUDA devices found");
}

int main() {
  const auto &CUDADevice = getCUDADevice();

  std::vector<char> DeviceBinary;
  loadDeviceBinary("simple_kernel", CUDADevice, DeviceBinary);

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
  for (int Idx = 0; Idx < GROUP_SIZE_X; ++Idx) {
    std::cout << "Data[" << Idx << "] = " << Data[Idx]
              << (Data[Idx] == Idx ? " (correct)" : " (incorrect)") << '\n';
  }

  OL_CHECK(olMemFree(Buffer));
  OL_CHECK(olDestroyProgram(Program));

  return 0;
}