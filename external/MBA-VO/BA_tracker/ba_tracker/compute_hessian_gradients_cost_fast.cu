#ifndef SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H
#define SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H

#include "compute_pixel_intensity.h"
#include "BA_tracker/core/common/SmallBlas.h"
#include "BA_tracker/core/common/Vector.h"
#include "reduction.h"
#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "BA_tracker/core/common/CustomType.h"

namespace SLAM
{
    namespace VO
    {

#define REDUCTION(size, buffer_, bid, tid)            \
    volatile float *buffer = buffer_;                 \
    assert(size < 65);                                \
    if (size == 64 && bid < 32)                       \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 32]; \
    }                                                 \
    if (size >= 32 && bid < 16)                       \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 16]; \
    }                                                 \
    if (size >= 16 && bid < 8)                        \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 8];  \
    }                                                 \
    if (size >= 8 && bid < 4)                         \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 4];  \
    }                                                 \
    if (size >= 4 && bid < 2)                         \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 2];  \
    }                                                 \
    if (size >= 2 && bid < 1)                         \
    {                                                 \
        buffer[tid] = buffer[tid] + buffer[tid + 1];  \
    }

        /** 
         *  This kernel computes the residual of "f(T) = 1/N sum_I_virtual (x) - I_blur(x)" 
         *  with respect to the 4 control knots; To lauch this kernel, we need to have 3 dimensional
         *  blocks, with dimension N_FRAMES x N_LOCALPATCHES x PATCH_SIZE. Each block has N threads.
         */
        __global__ void kernel_compute_pixel_jacobian_residual_fast(const unsigned char *I_ref,
                                                                    const float *dIxy_ref,
                                                                    unsigned char const *const *I_cur_imgs,
                                                                    const int num_vir_poses_per_frame,
                                                                    const int num_frames,
                                                                    const double *sampled_virtual_poses,
                                                                    const int spline_deg_k,
                                                                    const double *jacobian_virtual_pose_t_to_ctrl_knots,
                                                                    const double *jacobian_virtual_pose_R_to_ctrl_knots,
                                                                    const Core::Vector2d *local_patches_XY,
                                                                    const double *keypoints_z,
                                                                    const int num_keypoints,
                                                                    const int *local_patch_pattern_xy,
                                                                    const int patch_size,
                                                                    const float fx,
                                                                    const float fy,
                                                                    const float cx,
                                                                    const float cy,
                                                                    const int im_H,
                                                                    const int im_W,
                                                                    FLOAT *J_virtual_pixel_tR,
                                                                    FLOAT *virtual_pixel_residuals,
                                                                    double *pixel_jacobians_tR)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int num_blocks_per_virtual_view = (num_keypoints * patch_size) / blockDim.x + 1;
            const int global_pixel_idx = idx % (num_blocks_per_virtual_view * blockDim.x);

            if (global_pixel_idx >= num_keypoints * patch_size)
            {
                return;
            }

            const int virtualImgIdx = blockIdx.x / num_blocks_per_virtual_view;
            const int patchIdx = global_pixel_idx / patch_size;
            const int pixelIdx = global_pixel_idx % patch_size;

            // get virtual camera pose
            const double *t_c2r = sampled_virtual_poses + virtualImgIdx * 7;
            const double *R_c2r = t_c2r + 3;

            // get center local patches_xy
            const Core::Vector2d &patchXy = *(local_patches_XY + patchIdx);

            // get current pixel xy
            const int curPixelX = patchXy(0) + *(local_patch_pattern_xy + pixelIdx * 2);
            const int curPixelY = patchXy(1) + *(local_patch_pattern_xy + pixelIdx * 2 + 1);

            const int offset = virtualImgIdx * num_keypoints * patch_size + global_pixel_idx;

            // check if curXy is inside the image
            if (curPixelX < 0 || curPixelX > im_W - 1 || curPixelY < 0 || curPixelY > im_H - 1)
            {
                virtual_pixel_residuals[offset] = 0;
                return;
            }

            /**
            * COMPUTE PIXEL INTENSITY & JACOBIANS
            */
            const float qx = *R_c2r++;
            const float qy = *R_c2r++;
            const float qz = *R_c2r++;
            const float qw = *R_c2r++;
            const float x = *t_c2r++;
            const float y = *t_c2r++;
            const float z = *t_c2r++;

            // compute uray
            float x_hat = (curPixelX - cx) / fx;
            float y_hat = (curPixelY - cy) / fy;
            float z_hat = 1. / rsqrtf(1. + x_hat * x_hat + y_hat * y_hat);
            x_hat *= z_hat;
            y_hat *= z_hat;

            // compute lambda
            const float lambda = 2. * x_hat * (qx * qz - qw * qy) +
                                 2. * y_hat * (qx * qw + qy * qz) +
                                 z_hat * (qw * qw - qx * qx - qy * qy + qz * qz);

            const float depth = keypoints_z[patchIdx];
            const float DminusZoverLamda = (depth - z) / lambda;

            x_hat = DminusZoverLamda * x_hat;
            y_hat = DminusZoverLamda * y_hat;
            z_hat = DminusZoverLamda * z_hat;

            // compute ref_xy
            const float P3dx = x +
                               x_hat * (qw * qw + qx * qx - qy * qy - qz * qz) +
                               y_hat * (qx * qy - qw * qz) * 2 +
                               z_hat * (qw * qy + qx * qz) * 2;

            const float P3dy = y +
                               x_hat * (qw * qz + qx * qy) * 2 +
                               y_hat * (qw * qw - qx * qx + qy * qy - qz * qz) +
                               z_hat * (qy * qz - qw * qx) * 2;

            const float P3dz = z +
                               x_hat * (qx * qz - qw * qy) * 2 +
                               y_hat * (qw * qx + qy * qz) * 2 +
                               z_hat * (qw * qw - qx * qx - qy * qy + qz * qz);

            // normalize to image coordinate frame
            const float iz = 1.0f / (P3dz + 1e-8);
            const float ref_x = fx * P3dx * iz + cx;
            const float ref_y = fy * P3dy * iz + cy;

            if (ref_x < 0 || ref_x > im_W - 1 || ref_y < 0 || ref_y > im_H - 1)
            {
                virtual_pixel_residuals[offset] = 0;
                return;
            }

            // bilinear interpolation
            const int xi = ref_x;
            const int yi = ref_y;

            const float dx = ref_x - xi;
            const float dy = ref_y - yi;
            const float dxdy = dx * dy;

            const float w00 = 1.0f - dx - dy + dxdy;
            const float w01 = dx - dxdy;
            const float w10 = dy - dxdy;
            const float w11 = dxdy;

            const int index = yi * im_W + xi;
            const float interpolated_pixel_intensity = w00 * I_ref[index] +
                                                       w01 * I_ref[index + 1] +
                                                       w10 * I_ref[index + im_W] +
                                                       w11 * I_ref[index + im_W + 1];

            // get current image pixel intensity
            const unsigned char *I_cur = I_cur_imgs[0];
            const float intensity_cur = *(I_cur + curPixelY * im_W + curPixelX);
            virtual_pixel_residuals[offset] = (interpolated_pixel_intensity - intensity_cur) / num_vir_poses_per_frame;

            // compute jacobian 1x7
            if (pixel_jacobians_tR != nullptr)
            {
                const float dIx = w11 * dIxy_ref[2 * (index + im_W + 1)] +
                                  w10 * dIxy_ref[2 * (index + im_W)] +
                                  w01 * dIxy_ref[2 * (index + 1)] +
                                  w00 * dIxy_ref[2 * index];

                const float dIy = w11 * dIxy_ref[2 * (index + im_W + 1) + 1] +
                                  w10 * dIxy_ref[2 * (index + im_W) + 1] +
                                  w01 * dIxy_ref[2 * (index + 1) + 1] +
                                  w00 * dIxy_ref[2 * index + 1];

                const float T0 = qx * x_hat + qy * y_hat + qz * z_hat;
                const float T1 = qy * x_hat - qw * z_hat - qx * y_hat;
                const float T2 = qw * y_hat - qx * z_hat + qz * x_hat;
                const float T3 = qw * x_hat + qy * z_hat - qz * y_hat;
                const float T4 = qw * z_hat + qx * y_hat - qy * x_hat;

                const float C1 = 1. / (2.f * (x_hat * (-qw * qy + qx * qz) + y_hat * (qw * qx + qy * qz)) +
                                       z_hat * (qw * qw - qx * qx - qy * qy + qz * qz));

                const float K = -2.f * (qw * qz - qx * qy);
                const float L = 2.f * (qw * qy + qx * qz);
                const float M = 2.f * (qw * qz + qx * qy);
                const float N = -2.f * (qw * qx - qy * qz);
                const float P = -2.f * (qw * qy - qx * qz);
                const float Q = 2.f * (qw * qx + qy * qz);

                const float R0 = x_hat * (qw * qw + qx * qx - qy * qy - qz * qz) + y_hat * K + z_hat * L;
                const float R1 = x_hat * M + y_hat * (qw * qw - qx * qx + qy * qy - qz * qz) + z_hat * N;
                const float R2 = x_hat * P + y_hat * Q + z_hat * (qw * qw - qx * qx - qy * qy + qz * qz);

                //
                const float dPx_dqx = 2.f * (depth - z) * C1 * (T0 - T2 * C1 * R0);
                const float dPy_dqx = 2.f * (depth - z) * C1 * (T1 - T2 * C1 * R1);
                const float dPz_dqx = 2.f * (depth - z) * C1 * (T2 - T2 * C1 * R2);

                const float dPx_dqy = 2.f * (depth - z) * C1 * (T4 + T3 * C1 * R0);
                const float dPy_dqy = 2.f * (depth - z) * C1 * (T0 + T3 * C1 * R1);
                const float dPz_dqy = 2.f * (depth - z) * C1 * (T3 * C1 * R2 - T3);

                const float dPx_dqz = -2.f * (depth - z) * C1 * (T2 + T0 * C1 * R0);
                const float dPy_dqz = 2.f * (depth - z) * C1 * (T3 - T0 * C1 * R1);
                const float dPz_dqz = 2.f * (depth - z) * C1 * (T0 - T0 * C1 * R2);

                const float dPx_dqw = 2.f * (depth - z) * C1 * (T3 - T4 * C1 * R0);
                const float dPy_dqw = 2.f * (depth - z) * C1 * (T2 - T4 * C1 * R1);
                const float dPz_dqw = 2.f * (depth - z) * C1 * (T4 - T4 * C1 * R2);

                // projection jacobian
                const float dI_dPx = dIx * iz * fx;
                const float dI_dPy = dIy * iz * fy;
                const float dI_dPz = -iz * iz * (dIx * P3dx * fx + dIy * P3dy * fy);

                FLOAT *J_virtual_pixel_tR_ = J_virtual_pixel_tR + offset * 12;
                *(J_virtual_pixel_tR_++) = dI_dPx;
                *(J_virtual_pixel_tR_++) = dI_dPy;
                *(J_virtual_pixel_tR_++) = dI_dPz * (1.f - R2 * C1) - dI_dPx * R0 * C1 - dI_dPy * R1 * C1;
                *(J_virtual_pixel_tR_++) = dI_dPx * dPx_dqx + dI_dPy * dPy_dqx + dI_dPz * dPz_dqx;
                *(J_virtual_pixel_tR_++) = dI_dPx * dPx_dqy + dI_dPy * dPy_dqy + dI_dPz * dPz_dqy;
                *(J_virtual_pixel_tR_++) = dI_dPx * dPx_dqz + dI_dPy * dPy_dqz + dI_dPz * dPz_dqz;
                *(J_virtual_pixel_tR_++) = dI_dPx * dPx_dqw + dI_dPy * dPy_dqw + dI_dPz * dPz_dqw;
            }
        }

        __global__ void kernel_compute_pixel_jacobian_to_knottR(const int num_vir_poses_per_frame,
                                                                const int num_keypoints,
                                                                const int patch_size,
                                                                const double *jacobian_virtual_pose_t_to_ctrl_knots,
                                                                const double *jacobian_virtual_pose_R_to_ctrl_knots,
                                                                FLOAT *J_virtual_pixel_tR)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int num_blocks_per_virtual_view = (num_keypoints * patch_size) / blockDim.x + 1;
            const int global_pixel_idx = idx % (num_blocks_per_virtual_view * blockDim.x);

            const int virtual_img_idx = blockIdx.x / num_blocks_per_virtual_view;
            const int patchIdx = global_pixel_idx / patch_size;
            const int pixelIdx = global_pixel_idx % patch_size;

            // copy jacobian for virtual_view_pose to knots_tR to shared memory
            const double *jacobian_dPoset_dKnotst = jacobian_virtual_pose_t_to_ctrl_knots + virtual_img_idx * 18;
            const double *jacobian_dPoseR_dKnotsR = jacobian_virtual_pose_R_to_ctrl_knots + virtual_img_idx * 24;

            extern __shared__ float smem[];
            if (threadIdx.x < 18)
            {
                smem[threadIdx.x] = jacobian_dPoset_dKnotst[threadIdx.x];
            }
            if (threadIdx.x >= 18 && threadIdx.x < 42)
            {
                smem[threadIdx.x] = jacobian_dPoseR_dKnotsR[threadIdx.x - 18];
            }
            __syncthreads();

            // !!! This has to be placed here, such that we can pre-fetch data into 
            // !!! shared memory correctly.
            if (global_pixel_idx >= num_keypoints * patch_size)
            {
                return;
            }

            //
            const int offset0 = (virtual_img_idx * num_keypoints * patch_size + global_pixel_idx) * 12;
            FLOAT *J = J_virtual_pixel_tR + offset0;

            const float dI_dPose0 = J[0];
            const float dI_dPose1 = J[1];
            const float dI_dPose2 = J[2];
            const float dI_dPose3 = J[3];
            const float dI_dPose4 = J[4];
            const float dI_dPose5 = J[5];
            const float dI_dPose6 = J[6];

            const float *J_t = smem;
            const float *J_R = smem + 18;

            *(J++) = dI_dPose0 * J_t[0] + dI_dPose1 * J_t[6] + dI_dPose2 * J_t[12];
            *(J++) = dI_dPose0 * J_t[1] + dI_dPose1 * J_t[7] + dI_dPose2 * J_t[13];
            *(J++) = dI_dPose0 * J_t[2] + dI_dPose1 * J_t[8] + dI_dPose2 * J_t[14];
            *(J++) = dI_dPose0 * J_t[3] + dI_dPose1 * J_t[9] + dI_dPose2 * J_t[15];
            *(J++) = dI_dPose0 * J_t[4] + dI_dPose1 * J_t[10] + dI_dPose2 * J_t[16];
            *(J++) = dI_dPose0 * J_t[5] + dI_dPose1 * J_t[11] + dI_dPose2 * J_t[17];

            *(J++) = dI_dPose3 * J_R[0] + dI_dPose4 * J_R[6] + dI_dPose5 * J_R[12] + dI_dPose6 * J_R[18];
            *(J++) = dI_dPose3 * J_R[1] + dI_dPose4 * J_R[7] + dI_dPose5 * J_R[13] + dI_dPose6 * J_R[19];
            *(J++) = dI_dPose3 * J_R[2] + dI_dPose4 * J_R[8] + dI_dPose5 * J_R[14] + dI_dPose6 * J_R[20];
            *(J++) = dI_dPose3 * J_R[3] + dI_dPose4 * J_R[9] + dI_dPose5 * J_R[15] + dI_dPose6 * J_R[21];
            *(J++) = dI_dPose3 * J_R[4] + dI_dPose4 * J_R[10] + dI_dPose5 * J_R[16] + dI_dPose6 * J_R[22];
            *(J++) = dI_dPose3 * J_R[5] + dI_dPose4 * J_R[11] + dI_dPose5 * J_R[17] + dI_dPose6 * J_R[23];
        }

        __global__ void kernel_pixel_residual_reduction(const int num_vir_poses_per_frame,
                                                        const int num_keypoints,
                                                        const int patch_size,
                                                        const FLOAT *virtual_pixel_residuals,
                                                        double *pixel_residuals)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_keypoints * patch_size * num_vir_poses_per_frame)
            {
                return;
            }

            const int patchIdx = idx / (patch_size * num_vir_poses_per_frame);
            const int res = idx % (patch_size * num_vir_poses_per_frame);
            const int pixelIdx = res / num_vir_poses_per_frame;      // pixel index in the local patch
            const int virtualImgIdx = res % num_vir_poses_per_frame; // image index from the N local virtual images
            const int global_pixel_idx = patchIdx * patch_size + pixelIdx;
            const int index = virtualImgIdx * num_keypoints * patch_size + global_pixel_idx;

            extern __shared__ float smem[];
            smem[threadIdx.x] = virtual_pixel_residuals[index];
            __syncthreads();

            REDUCTION(num_vir_poses_per_frame, smem, virtualImgIdx, threadIdx.x);

            if (virtualImgIdx == 0)
            {
                pixel_residuals[global_pixel_idx] = smem[threadIdx.x];
            }
        }

        __global__ void kernel_pixel_jacobian_reduction(const int num_vir_poses_per_frame,
                                                        const int num_keypoints,
                                                        const int patch_size,
                                                        const FLOAT *J_virtual_pixel_tR,
                                                        double *pixel_jacobians_tR)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= num_keypoints * patch_size * num_vir_poses_per_frame)
            {
                return;
            }

            const int patchIdx = idx / (patch_size * num_vir_poses_per_frame);
            const int res = idx % (patch_size * num_vir_poses_per_frame);
            const int pixelIdx = res / num_vir_poses_per_frame;      // pixel index in the local patch
            const int virtualImgIdx = res % num_vir_poses_per_frame; // image index from the N local virtual images
            const int global_pixel_idx = patchIdx * patch_size + pixelIdx;

            const int index = 12 * (virtualImgIdx * num_keypoints * patch_size + global_pixel_idx);

            extern __shared__ float smem[];
            smem[threadIdx.x] = J_virtual_pixel_tR[index];
            smem[threadIdx.x + blockDim.x] = J_virtual_pixel_tR[index + 1];
            smem[threadIdx.x + 2 * blockDim.x] = J_virtual_pixel_tR[index + 2];
            smem[threadIdx.x + 3 * blockDim.x] = J_virtual_pixel_tR[index + 3];
            smem[threadIdx.x + 4 * blockDim.x] = J_virtual_pixel_tR[index + 4];
            smem[threadIdx.x + 5 * blockDim.x] = J_virtual_pixel_tR[index + 5];
            smem[threadIdx.x + 6 * blockDim.x] = J_virtual_pixel_tR[index + 6];
            smem[threadIdx.x + 7 * blockDim.x] = J_virtual_pixel_tR[index + 7];
            smem[threadIdx.x + 8 * blockDim.x] = J_virtual_pixel_tR[index + 8];
            smem[threadIdx.x + 9 * blockDim.x] = J_virtual_pixel_tR[index + 9];
            smem[threadIdx.x + 10 * blockDim.x] = J_virtual_pixel_tR[index + 10];
            smem[threadIdx.x + 11 * blockDim.x] = J_virtual_pixel_tR[index + 11];
            __syncthreads();

            {
                REDUCTION(num_vir_poses_per_frame, smem, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 2 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 3 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 4 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 5 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 6 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 7 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 8 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 9 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 10 * blockDim.x, virtualImgIdx, threadIdx.x);
            }
            {
                REDUCTION(num_vir_poses_per_frame, smem + 11 * blockDim.x, virtualImgIdx, threadIdx.x);
            }

            if (virtualImgIdx == 0)
            {
                float scale = 1.f / num_vir_poses_per_frame;
                pixel_jacobians_tR[global_pixel_idx * 12] = smem[threadIdx.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 1] = smem[threadIdx.x + blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 2] = smem[threadIdx.x + 2 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 3] = smem[threadIdx.x + 3 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 4] = smem[threadIdx.x + 4 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 5] = smem[threadIdx.x + 5 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 6] = smem[threadIdx.x + 6 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 7] = smem[threadIdx.x + 7 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 8] = smem[threadIdx.x + 8 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 9] = smem[threadIdx.x + 9 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 10] = smem[threadIdx.x + 10 * blockDim.x] * scale;
                pixel_jacobians_tR[global_pixel_idx * 12 + 11] = smem[threadIdx.x + 11 * blockDim.x] * scale;
            }
        }

        void compute_pixel_jacobian_residual_fast(const unsigned char *I_ref,
                                                  const float *dIxy_ref,
                                                  unsigned char const *const *I_cur_imgs,
                                                  const int num_vir_poses_per_frame,
                                                  const int num_frames,
                                                  const double *sampled_virtual_poses,
                                                  const int spline_deg_k,
                                                  const double *jacobian_virtual_pose_t_to_ctrl_knots,
                                                  const double *jacobian_virtual_pose_R_to_ctrl_knots,
                                                  const Core::Vector2d *local_patches_XY,
                                                  const double *keypoints_z,
                                                  const int num_keypoints,
                                                  const int *local_patch_pattern_xy,
                                                  const int patch_size,
                                                  const Core::VectorX<double, 4> &intrinsics,
                                                  const Core::VectorX<int, 2> &im_size_HW,
                                                  FLOAT *J_virtual_pixel_tR,
                                                  FLOAT *virtual_pixel_residuals,
                                                  double *pixel_residuals,
                                                  double *pixel_jacobians_tR)
        {
            assert(num_frames == 1);
            assert(spline_deg_k == 2);

            const int num_threads_per_block = 128;
            const int num_blocks_per_virtual_view = (num_keypoints * patch_size) / num_threads_per_block + 1;
            const int total_num_blocks = num_blocks_per_virtual_view * num_vir_poses_per_frame;
            dim3 gridDim = dim3(total_num_blocks);
            dim3 blockDim = dim3(num_threads_per_block);

            const float fx = intrinsics.values[0];
            const float fy = intrinsics.values[1];
            const float cx = intrinsics.values[2];
            const float cy = intrinsics.values[3];
            const int H = im_size_HW.values[0];
            const int W = im_size_HW.values[1];

            kernel_compute_pixel_jacobian_residual_fast<<<gridDim, blockDim>>>(
                I_ref,
                dIxy_ref,
                I_cur_imgs,
                num_vir_poses_per_frame,
                num_frames,
                sampled_virtual_poses,
                spline_deg_k,
                jacobian_virtual_pose_t_to_ctrl_knots,
                jacobian_virtual_pose_R_to_ctrl_knots,
                local_patches_XY,
                keypoints_z,
                num_keypoints,
                local_patch_pattern_xy,
                patch_size,
                fx,
                fy,
                cx,
                cy,
                H,
                W,
                J_virtual_pixel_tR,
                virtual_pixel_residuals,
                pixel_jacobians_tR);
            cudaDeviceSynchronize();

            gridDim = dim3(num_keypoints * patch_size * num_vir_poses_per_frame / num_threads_per_block + 1);
            blockDim = dim3(num_threads_per_block);
            kernel_pixel_residual_reduction<<<gridDim, blockDim, sizeof(float) * num_threads_per_block>>>(
                num_vir_poses_per_frame,
                num_keypoints,
                patch_size,
                virtual_pixel_residuals,
                pixel_residuals);
            cudaDeviceSynchronize();

            if (pixel_jacobians_tR != nullptr)
            {
                dim3 gridDim = dim3(total_num_blocks);
                dim3 blockDim = dim3(num_threads_per_block);
                kernel_compute_pixel_jacobian_to_knottR<<<gridDim, blockDim, sizeof(float) * 42>>>(
                    num_vir_poses_per_frame,
                    num_keypoints,
                    patch_size,
                    jacobian_virtual_pose_t_to_ctrl_knots,
                    jacobian_virtual_pose_R_to_ctrl_knots,
                    J_virtual_pixel_tR);
                cudaDeviceSynchronize();

                gridDim = dim3(num_keypoints * patch_size * num_vir_poses_per_frame / num_threads_per_block + 1);
                blockDim = dim3(num_threads_per_block);
                kernel_pixel_jacobian_reduction<<<gridDim, blockDim, sizeof(float) * num_threads_per_block * 12>>>(
                    num_vir_poses_per_frame,
                    num_keypoints,
                    patch_size,
                    J_virtual_pixel_tR,
                    pixel_jacobians_tR);
            }

            cudaDeviceSynchronize();
        }

#undef REDUCTION
    } // namespace VO
} // namespace SLAM

#endif