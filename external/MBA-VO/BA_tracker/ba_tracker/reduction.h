#ifndef SLAM_VO_REDUCTION_H
#define SLAM_VO_REDUCTION_H

#include "BA_tracker/core/common/CudaDefs.h"
#include <cstdio>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

namespace SLAM
{
    namespace VO
    {
        __device__ __forceinline__ void reduce(volatile double *buffer, const int size)
        {
            const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
            double value = buffer[thread_id];

            if (size >= 1024)
            {
                if (thread_id < 512)
                    buffer[thread_id] = value = value + buffer[thread_id + 512];
                __syncthreads();
            }
            if (size >= 512)
            {
                if (thread_id < 256)
                    buffer[thread_id] = value = value + buffer[thread_id + 256];
                __syncthreads();
            }
            if (size >= 256)
            {
                if (thread_id < 128)
                    buffer[thread_id] = value = value + buffer[thread_id + 128];
                __syncthreads();
            }
            if (size >= 128)
            {
                if (thread_id < 64)
                    buffer[thread_id] = value = value + buffer[thread_id + 64];
                __syncthreads();
            }

            if (size >= 64 && thread_id < 32)
                buffer[thread_id] = value = value + buffer[thread_id + 32];
            if (size >= 32 && thread_id < 16)
                buffer[thread_id] = value = value + buffer[thread_id + 16];
            if (size >= 16 && thread_id < 8)
                buffer[thread_id] = value = value + buffer[thread_id + 8];
            if (size >= 8 && thread_id < 4)
                buffer[thread_id] = value = value + buffer[thread_id + 4];
            if (size >= 4 && thread_id < 2)
                buffer[thread_id] = value = value + buffer[thread_id + 2];
            if (size >= 2 && thread_id < 1)
                buffer[thread_id] = value = value + buffer[thread_id + 1];
        } // namespace VO
    }     // namespace VO
} // namespace SLAM

#endif