#include <gpuintrin.h>
#include <stdio.h>

extern "C" __gpu_kernel void printHelloWorld() {
  printf("GPU block %d and thread %d\n", __gpu_block_id(__GPU_X_DIM),
         __gpu_thread_id(__GPU_X_DIM));
}