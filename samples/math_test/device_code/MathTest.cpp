#include <gpuintrin.h>
#include <math.h>
#include <stdlib.h>

extern "C" __gpu_kernel void applyLogf(const float *In, float *Out,
                                       size_t NumElements) {
  int Index = __gpu_num_threads(__GPU_X_DIM) * __gpu_block_id(__GPU_X_DIM) +
              __gpu_thread_id(__GPU_X_DIM);

  if (Index < NumElements) {
    Out[Index] = logf(In[Index]);
  }
}