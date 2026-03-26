#ifndef SLAM_CORE_CUDA_ARITHMETIC_CUH_
#define SLAM_CORE_CUDA_ARITHMETIC_CUH_

#include "BA_tracker/core/common/CudaDefs.h"

namespace SLAM
{
    namespace Core
    {
        __global__ void kernel_elem_addition_cuh(double *a, double *b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] + b[idx];
        }

        __forceinline__ void cuda_elem_addition_cuh(double *a, double *b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_addition_cuh<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

    } // namespace Core
} // namespace SLAM

#endif