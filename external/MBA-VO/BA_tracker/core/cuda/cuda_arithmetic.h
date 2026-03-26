#ifndef CORE_CUDA_CUDA_ARITHMETIC_H_
#define CORE_CUDA_CUDA_ARITHMETIC_H_

#include "BA_tracker/core/common/CudaDefs.h"

namespace SLAM
{
    namespace Core
    {
        void cuda_truncation(double *value, int n, double min, double max);

        void cuda_addition(double *a, double b, double *c, int n);

        void cuda_inverse(double *a, double *b, int n);

        void cuda_multiply(double *values_in, int n_elems, double multiplier, double *values_out);
        void cuda_dot(double *a, double *b, int N, double *product);

        void cuda_elem_addition(double *a, double *b, double *c, int n);
        void cuda_elem_substraction(double *a, double *b, double *c, int n);
        void cuda_elem_multiply(double *a, double *b, double *c, int n);
        void cuda_elem_division(double *a, double *b, double *c, int n);
        void cuda_elem_sqrt(double *a, double *b, int n);
    } // namespace Core
} // namespace SLAM

#endif