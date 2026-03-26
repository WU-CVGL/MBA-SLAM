#ifndef CORE_COMMON_VECTOR_H_
#define CORE_COMMON_VECTOR_H_

#include "CudaDefs.h"
#include <cmath>

namespace SLAM
{
    namespace Core
    {
        template <class T, int nDim_>
        struct VectorX
        {
            int nDim;
            T values[nDim_];
        };

        struct Vector2d : public VectorX<double, 2>
        {
            __CPU_AND_CUDA_CODE__ Vector2d()
            {
                this->nDim = 2;
            }

            __CPU_AND_CUDA_CODE__ Vector2d(double x, double y)
            {
                this->nDim = 2;
                values[0] = x;
                values[1] = y;
            }

            __CPU_AND_CUDA_CODE__ double &operator()(int i)
            {
                return *(this->values + i);
            }

            __CPU_AND_CUDA_CODE__ double operator()(int i) const
            {
                return *(this->values + i);
            }

            __CPU_AND_CUDA_CODE__ Vector2d operator+(const Vector2d &b) const
            {
                Vector2d c;
                c(0) = this->values[0] + b(0);
                c(1) = this->values[1] + b(1);
                return c;
            }

            __CPU_AND_CUDA_CODE__ Vector2d operator-(const Vector2d &b) const
            {
                Vector2d c;
                c(0) = this->values[0] - b(0);
                c(1) = this->values[1] - b(1);
                return c;
            }

            __CPU_AND_CUDA_CODE__ Vector2d operator-() const
            {
                Vector2d c;
                c(0) = -this->values[0];
                c(1) = -this->values[1];
                return c;
            }

            __CPU_AND_CUDA_CODE__ double norm() const
            {
                return sqrt(this->values[0] * this->values[0] + this->values[1] * this->values[1]);
            }
        };

        struct Vector3d : public VectorX<double, 3>
        {
            __CPU_AND_CUDA_CODE__ Vector3d()
            {
                this->nDim = 3;
            }

            __CPU_AND_CUDA_CODE__ Vector3d(double x, double y, double z)
            {
                this->nDim = 3;
                values[0] = x;
                values[1] = y;
                values[2] = z;
            }

            __CPU_AND_CUDA_CODE__ double &operator()(int i)
            {
                return *(this->values + i);
            }

            __CPU_AND_CUDA_CODE__ double operator()(int i) const
            {
                return *(this->values + i);
            }

            __CPU_AND_CUDA_CODE__ Vector3d operator+(const Vector3d &b) const
            {
                Vector3d c;
                c(0) = this->values[0] + b(0);
                c(1) = this->values[1] + b(1);
                c(2) = this->values[2] + b(2);
                return c;
            }

            __CPU_AND_CUDA_CODE__ Vector3d operator-(const Vector3d &b) const
            {
                Vector3d c;
                c(0) = this->values[0] - b(0);
                c(1) = this->values[1] - b(1);
                c(2) = this->values[2] - b(2);
                return c;
            }

            __CPU_AND_CUDA_CODE__ Vector3d operator-() const
            {
                Vector3d c;
                c(0) = -this->values[0];
                c(1) = -this->values[1];
                c(2) = -this->values[2];
                return c;
            }

            __CPU_AND_CUDA_CODE__ Vector3d operator*(double b) const
            {
                Vector3d c;
                c(0) = this->values[0] * b;
                c(1) = this->values[1] * b;
                c(2) = this->values[2] * b;
                return c;
            }

            __CPU_AND_CUDA_CODE__ double norm() const
            {
                return sqrt(this->values[0] * this->values[0] +
                            this->values[1] * this->values[1] +
                            this->values[2] * this->values[2]);
            }

            __CPU_AND_CUDA_CODE__ double squaredNorm() const
            {
                return this->values[0] * this->values[0] +
                       this->values[1] * this->values[1] +
                       this->values[2] * this->values[2];
            }
        };
    } // namespace Core
} // namespace SLAM

#endif