#ifndef SRC_CORE_COMMON_CUDA_DEFS
#define SRC_CORE_COMMON_CUDA_DEFS

#ifdef COMPILE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#define __CPU_AND_CUDA_CODE__ __device__ __host__
#define __FORCEINLINE__ __forceinline__
#else
#define __CPU_AND_CUDA_CODE__
#define __FORCEINLINE__ inline
#define __device__
#endif

#endif