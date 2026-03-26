#ifndef SLAM_CORE_CUDA_REDUCTION_H_
#define SLAM_CORE_CUDA_REDUCTION_H_

#include "BA_tracker/core/common/CudaDefs.h"
#include <cub/cub.cuh>

namespace SLAM
{
    namespace Core
    {
        // inline functions
        template <typename T>
        __global__ void kernel_hwc_to_chw(T *d_in, T *d_out, int H, int W, int C)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= H * W * C)
            {
                return;
            }

            const int c = i % C;
            const int p = i / C;
            const int w = p % W;
            const int h = p / W;
            d_out[i] = d_in[c * H * W + h * W + w];
        }

        template <typename T>
        __global__ void kernel_col_major_to_row_major(T *d_in, T *d_out, int H, int W)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= H * W)
            {
                return;
            }
            const int x = i % W;
            const int y = i / W;
            *(d_out + y * W + x) = *(d_in + x * H + y);
        }

        inline void hwc_to_chw(double *d_in, double *d_out, int H, int W, int C)
        {
            int nThreads = H * W * C;
            dim3 gridDim = dim3(nThreads / 1024 + 1);
            dim3 blockDim = dim3(1024);
            kernel_hwc_to_chw<double><<<gridDim, blockDim>>>(d_in, d_out, H, W, C);
            cudaDeviceSynchronize();
        }

        inline void col_major_to_row_major(double *d_in, double *d_out, int H, int W)
        {
            int nThreads = H * W;
            dim3 gridDim = dim3(nThreads / 1024 + 1);
            dim3 blockDim = dim3(1024);
            kernel_col_major_to_row_major<double><<<gridDim, blockDim>>>(d_in, d_out, H, W);
            cudaDeviceSynchronize();
        }

        inline void cuda_sum_reduction(double *data, int num_elems, double *sum)
        {
            void *temp_storage = NULL;
            size_t temp_storage_bytes = 0;

            cub::DeviceReduce::Sum(temp_storage,
                                   temp_storage_bytes,
                                   data,
                                   sum,
                                   num_elems);

            cudaMalloc(&temp_storage, temp_storage_bytes);

            cub::DeviceReduce::Sum(temp_storage,
                                   temp_storage_bytes,
                                   data,
                                   sum,
                                   num_elems);
        }

        inline void cuda_channel_wise_sum_reduction(double *data, int H, int W, int C, double *sum)
        {
            // transform data memory layout from HWC to CHW
            double *transformed_data;
            cudaMalloc(&transformed_data, sizeof(double) * H * W * C);
            hwc_to_chw(data, transformed_data, H, W, C);

            int HW = H * W;
            for (int i = 0; i < HW; i++)
            {
                cuda_sum_reduction(transformed_data + i * C, C, sum + i);
            }

            cudaFree(transformed_data);
        }

        inline void cuda_row_reduction(double *col_major_data, int H, int W, double *sum)
        {
            double *row_major_data;
            cudaMalloc(&row_major_data, sizeof(double) * H * W);
            col_major_to_row_major(col_major_data, row_major_data, H, W);
            for (int i = 0; i < H; i++)
            {
                cuda_sum_reduction(row_major_data + i * W, W, sum + i);
            }
            cudaFree(row_major_data);
        }
    } // namespace Core
} // namespace SLAM

#endif
