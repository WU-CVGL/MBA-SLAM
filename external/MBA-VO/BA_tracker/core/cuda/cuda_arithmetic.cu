#include <BA_tracker/core/common/CudaAtomics.h>
#include <BA_tracker/core/common/CudaDefs.h>
#include "cuda_arithmetic.h"
#include "cuda_reduction.h"

namespace SLAM
{
    namespace Core
    {
        __global__ void kernel_truncation(double *values, int n_elems, double min, double max)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_elems)
            {
                return;
            }
            values[idx] = fmaxf(fminf(values[idx], max), min);
        }

        __global__ void kernel_inverse(double *a, double *b, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            b[idx] = 1. / a[idx];
        }

        __global__ void kernel_addition(double *a, double b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] + b;
        }

        __global__ void kernel_multiply(double *values_in, int n_elems, double multiplier, double *values_out)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_elems)
            {
                return;
            }
            values_out[idx] = values_in[idx] * multiplier;
        }

        __global__ void kernel_elem_addition(double *a, double *b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] + b[idx];
        }

        __global__ void kernel_elem_substraction(double *a, double *b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] - b[idx];
        }

        __global__ void kernel_elem_multiply(double *a, double *b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] * b[idx];
        }

        __global__ void kernel_elem_division(double *a, double *b, double *c, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            c[idx] = a[idx] /(1e-8+ b[idx]);
        }

        __global__ void kernel_elem_sqrt(double *a, double *b, int n)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
            {
                return;
            }
            b[idx] = sqrt(a[idx]);
        }

        __global__ void kernel_dot(double *a, double *b, int N, double *elem_product)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N)
            {
                return;
            }

            elem_product[i] = a[i] * b[i];
        }

        void cuda_truncation(double *values, int n_elems, double min, double max)
        {
            int nThreads = n_elems;
            dim3 gridDim = dim3(nThreads / 32 + 1);
            dim3 blockDim = dim3(32);
            kernel_truncation<<<gridDim, blockDim>>>(values, n_elems, min, max);
            cudaDeviceSynchronize();
        }

        void cuda_addition(double *a, double b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_addition<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

        void cuda_inverse(double *a, double *b, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_inverse<<<gridDim, blockDim>>>(a, b, n);
            cudaDeviceSynchronize();
        }

        void cuda_multiply(double *values_in, int n_elems, double multiplier, double *values_out)
        {
            int nThreads = n_elems;
            dim3 gridDim = dim3(nThreads / 32 + 1);
            dim3 blockDim = dim3(32);
            kernel_multiply<<<gridDim, blockDim>>>(values_in, n_elems, multiplier, values_out);
            cudaDeviceSynchronize();
        }

        void cuda_elem_addition(double *a, double *b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_addition<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

        void cuda_elem_substraction(double *a, double *b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_substraction<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

        void cuda_elem_multiply(double *a, double *b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_multiply<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

        void cuda_elem_division(double *a, double *b, double *c, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_division<<<gridDim, blockDim>>>(a, b, c, n);
            cudaDeviceSynchronize();
        }

        void cuda_elem_sqrt(double *a, double *b, int n)
        {
            int nThreads = n;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);
            kernel_elem_sqrt<<<gridDim, blockDim>>>(a, b, n);
            cudaDeviceSynchronize();
        }

        void cuda_dot(double *a, double *b, int N, double *product)
        {
            int nThreads = N;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);

            double* elem_product;
            cudaMalloc(&elem_product, sizeof(double) * N);

            kernel_dot<<<gridDim, blockDim>>>(a, b, N, elem_product);
            cudaDeviceSynchronize();
            
            cuda_sum_reduction(elem_product, N, product);
            cudaFree(elem_product);
        }
    } // namespace Core
} // namespace SLAM