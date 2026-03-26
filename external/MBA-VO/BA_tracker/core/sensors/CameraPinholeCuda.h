#ifndef CORE_SENSORS_CAMERA_PINHOLE_CUDA_H_
#define CORE_SENSORS_CAMERA_PINHOLE_CUDA_H_

#include "BA_tracker/core/common/CudaDefs.h"
#include "BA_tracker/core/common/Matrix.h"
#include "BA_tracker/core/common/Quaternion.h"
#include "BA_tracker/core/common/Vector.h"
#include <stdio.h>

namespace SLAM
{
    namespace Core
    {
        struct CameraPinholeFunctor
        {
            __CPU_AND_CUDA_CODE__ bool project(const Vector3d &P3d, Vector2d &p2d)
            {
                if (P3d(2) < 0)
                {
                    return false;
                }
                p2d(0) = mfx * P3d(0) / P3d(2) + mcx;
                p2d(1) = mfy * P3d(1) / P3d(2) + mcy;
                return true;
            }

            __CPU_AND_CUDA_CODE__ bool unproject(const Vector2d &p2d, double d, Vector3d &P3d)
            {
                double p2dn_x = (p2d(0) - mcx) / mfx;
                double p2dn_y = (p2d(1) - mcy) / mfy;

                P3d(0) = p2dn_x * d;
                P3d(1) = p2dn_y * d;
                P3d(2) = d;

                return true;
            }

            __CPU_AND_CUDA_CODE__ void projection_jacobian(const Vector3d &P3d, MatrixXX<double, 2, 3> &jacobian)
            {
                double iz = 1.0f / P3d(2);
                double iz2 = iz * iz;

                jacobian(0, 0) = iz * mfx;
                jacobian(0, 1) = 0.0f;
                jacobian(0, 2) = (-1.0f) * P3d(0) * iz2 * mfx;
                jacobian(1, 0) = 0.0f;
                jacobian(1, 1) = iz * mfy;
                jacobian(1, 2) = (-1.0f) * P3d(1) * iz2 * mfy;
            }

            // internal data
            int mH, mW;
            double mfx, mfy, mcx, mcy;

            // T_b2s = [mR_b2s | mt_b2s]
            Quaterniond mR_b2s;
            Vector3d mt_b2s;
        };

        class CameraPinholeCpuCuda
        {
        public:
            CameraPinholeCpuCuda(int H, int W, double fx, double fy, double cx, double cy)
                : camera_cpu_(nullptr),
                  camera_cuda_(nullptr)
            {
                camera_cpu_ = new CameraPinholeFunctor();
                camera_cpu_->mH = H;
                camera_cpu_->mW = W;
                camera_cpu_->mfx = fx;
                camera_cpu_->mfy = fy;
                camera_cpu_->mcx = cx;
                camera_cpu_->mcy = cy;
            }

            CameraPinholeCpuCuda(const Quaterniond &R_b2s, const Vector3d &t_b2s, int H, int W, double fx, double fy, double cx, double cy)
                : camera_cpu_(nullptr),
                  camera_cuda_(nullptr)
            {
                camera_cpu_ = new CameraPinholeFunctor();
                camera_cpu_->mR_b2s = R_b2s;
                camera_cpu_->mt_b2s = t_b2s;
                camera_cpu_->mH = H;
                camera_cpu_->mW = W;
                camera_cpu_->mfx = fx;
                camera_cpu_->mfy = fy;
                camera_cpu_->mcx = cx;
                camera_cpu_->mcy = cy;
            }

            ~CameraPinholeCpuCuda()
            {
                delete camera_cpu_;
#ifdef COMPILE_WITH_CUDA
                cudaFree(camera_cuda_);
#endif
            }

        public:
            CameraPinholeFunctor *get_functor(bool use_cuda)
            {
                if (use_cuda)
                {
                    return camera_cuda_;
                }
                else
                {
                    return camera_cpu_;
                }
            }

            void allocate_cuda_functor()
            {
                if (camera_cuda_ == nullptr)
                {
                    cudaMalloc((void **)&camera_cuda_, sizeof(CameraPinholeFunctor));
                }
            }

            void upload_data_to_device()
            {
#ifdef COMPILE_WITH_CUDA
                // create camera
                Quaterniond R_b2c = camera_cpu_->mR_b2s;
                Vector3d t_b2c = camera_cpu_->mt_b2s;
                int H = camera_cpu_->mH;
                int W = camera_cpu_->mW;
                double fx = camera_cpu_->mfx;
                double cx = camera_cpu_->mcx;
                double fy = camera_cpu_->mfy;
                double cy = camera_cpu_->mcy;

                // create CUDA camera
                cudaMemcpy(&camera_cuda_->mR_b2s.x, &R_b2c.x, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mR_b2s.y, &R_b2c.y, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mR_b2s.z, &R_b2c.z, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mR_b2s.w, &R_b2c.w, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(camera_cuda_->mt_b2s.values, t_b2c.values, sizeof(double) * 3, cudaMemcpyHostToDevice);

                cudaMemcpy(&camera_cuda_->mH, &H, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mW, &W, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mfx, &fx, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mfy, &fy, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mcx, &cx, sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(&camera_cuda_->mcy, &cy, sizeof(double), cudaMemcpyHostToDevice);
#endif
            }

        private:
            CameraPinholeFunctor *camera_cpu_;
            CameraPinholeFunctor *camera_cuda_;
        };
    } // namespace Core
} // namespace SLAM

#endif