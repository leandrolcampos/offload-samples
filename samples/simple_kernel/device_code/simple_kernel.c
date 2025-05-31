#include <gpuintrin.h>

__gpu_kernel void simple_kernel(int *out) {
  out[__gpu_thread_id(0)] = __gpu_thread_id(0);
}