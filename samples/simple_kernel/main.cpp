#include "OffloadUtils.hpp"
#include <iostream>
#include <vector>

#define GROUP_SIZE_X 8

int main() {
  const auto &CUDADevice = ols::getCUDADevice();

  std::vector<char> DeviceBinary;
  loadDeviceBinary("SimpleKernel", CUDADevice, DeviceBinary);

  ol_program_handle_t Program = nullptr;
  OL_CHECK(olCreateProgram(CUDADevice.Handle, DeviceBinary.data(),
                           DeviceBinary.size(), &Program));

  ol_kernel_handle_t Kernel = nullptr;
  OL_CHECK(olGetKernel(Program, "simpleKernel", &Kernel));

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