#include <gpuintrin.h>

extern "C" __gpu_kernel void simpleKernel(int *Out) {
  Out[__gpu_thread_id(__GPU_X_DIM)] = __gpu_thread_id(__GPU_X_DIM);
}