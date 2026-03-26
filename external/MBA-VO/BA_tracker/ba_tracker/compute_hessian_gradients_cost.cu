#ifndef SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H
#define SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H

#include "compute_pixel_intensity.h"
#include "BA_tracker/core/common/CustomType.h"
#include "BA_tracker/core/common/SmallBlas.h"
#include "BA_tracker/core/common/Vector.h"
#include "reduction.h"
#include <assert.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace SLAM
{
    namespace VO
    {
        /** 
         *  This kernel computes the residual of "f(T) = 1/N sum_I_virtual (x) - I_blur(x)" 
         *  with respect to the 4 control knots; To lauch this kernel, we need to have 3 dimensional
         *  blocks, with dimension N_FRAMES x N_LOCALPATCHES x PATCH_SIZE. Each block has N threads.
         */
        __global__ void kernel_compute_pixel_jacobian_residual(const unsigned char *I_ref,
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
                                                               const double fx,
                                                               const double fy,
                                                               const double cx,
                                                               const double cy,
                                                               const Core::VectorX<int, 2> im_size_HW,
                                                               FLOAT *jacobian_pixel_to_ctrl_knots_tR,
                                                               double *pixel_residuals,
                                                               double *pixel_jacobians_tR)
        {
            assert(gridDim.x == num_frames);
            assert(gridDim.y == num_keypoints);
            assert(gridDim.z == patch_size);
            assert(blockDim.x == num_vir_poses_per_frame);

            const int frameIdx = blockIdx.x;
            const int patchIdx = blockIdx.y;
            const int pixelIdx = blockIdx.z;       // pixel index in the local patch
            const int virtualImgIdx = threadIdx.x; // image index from the N local virtual images
            const int global_pixel_idx = frameIdx * num_keypoints * patch_size + patchIdx * patch_size + pixelIdx;

            // get virtual camera pose
            const int globalVirtualImgIdx = frameIdx * num_vir_poses_per_frame + virtualImgIdx;
            const double *t_c2r = sampled_virtual_poses + globalVirtualImgIdx * 7;
            const double *R_c2r = t_c2r + 3;

            // get center local patches_xy
            const int globalPatchIdx = frameIdx * num_keypoints + patchIdx;
            const Core::Vector2d patchXy = *(local_patches_XY + globalPatchIdx);

            // get current pixel xy
            const int dx = *(local_patch_pattern_xy + pixelIdx * 2);
            const int dy = *(local_patch_pattern_xy + pixelIdx * 2 + 1);
            const int curPixelX = patchXy(0) + dx;
            const int curPixelY = patchXy(1) + dy;
            const Core::Vector2d curPixelXy(curPixelX, curPixelY);

            // check if curXy is inside the image
            if (curPixelX < 0 || curPixelX > im_size_HW.values[1] - 1 || curPixelY < 0 || curPixelY > im_size_HW.values[0] - 1)
            {
                pixel_residuals[global_pixel_idx] = 0;
                return;
            }

            // get interpolated pixel intensity & jacobian
            double interpolated_pixel_intensity;
            double dI_dPose[7];
            double *J = dI_dPose;
            if (pixel_jacobians_tR == nullptr)
            {
                J = nullptr;
            }

            extern __shared__ double smem[];
            const int tid = threadIdx.y * blockDim.x + threadIdx.x;
            smem[tid] = 0;

            if (!compute_pixel_intensity<double>(I_ref,
                                                 dIxy_ref,
                                                 im_size_HW.values[0],
                                                 im_size_HW.values[1],
                                                 R_c2r,
                                                 t_c2r,
                                                 keypoints_z[patchIdx],
                                                 fx,
                                                 fy,
                                                 cx,
                                                 cy,
                                                 curPixelXy,
                                                 &interpolated_pixel_intensity,
                                                 J))
            {
                pixel_residuals[global_pixel_idx] = 0;
                return;
            }

            smem[tid] = interpolated_pixel_intensity;
            __syncthreads();
            reduce(smem, blockDim.x);
            if (threadIdx.x == 0)
            {
                // get current image pixel intensity
                const unsigned char *I_cur = I_cur_imgs[frameIdx];
                const double intensity_cur = *(I_cur + curPixelY * im_size_HW.values[1] + curPixelX);
                pixel_residuals[global_pixel_idx] = smem[0] / float(num_vir_poses_per_frame) - intensity_cur;
            }

            if (pixel_jacobians_tR != nullptr)
            {
                // TODO: should we move this to global memory?
                // compute dI_dCtrlKnots: i.e., 1x24
                int idx = global_pixel_idx * num_vir_poses_per_frame + virtualImgIdx;
                FLOAT *dI_dCtrlKnots_tR = jacobian_pixel_to_ctrl_knots_tR + idx * 6 * spline_deg_k;

                const double *jacobian_dPoset_dKnotst = jacobian_virtual_pose_t_to_ctrl_knots +
                                                        globalVirtualImgIdx * 3 * 3 * spline_deg_k;

                const double *jacobian_dPoseR_dKnotsR = jacobian_virtual_pose_R_to_ctrl_knots +
                                                        globalVirtualImgIdx * 4 * 3 * spline_deg_k;

                Core::MatrixMatrixMultiply<double, double, FLOAT, 0>(dI_dPose, 1, 3,
                                                                     jacobian_dPoset_dKnotst, 3, 3 * spline_deg_k,
                                                                     dI_dCtrlKnots_tR, 0, 0, 1, 3 * spline_deg_k);

                Core::MatrixMatrixMultiply<double, double, FLOAT, 0>(dI_dPose + 3, 1, 4,
                                                                     jacobian_dPoseR_dKnotsR, 4, 3 * spline_deg_k,
                                                                     dI_dCtrlKnots_tR, 0, 3 * spline_deg_k, 1, 3 * spline_deg_k);

                // do sum reduction for both blurred pixel intensity and jacobian
                for (int i = 0; i < 6 * spline_deg_k; i++)
                {
                    smem[tid] = dI_dCtrlKnots_tR[i];
                    __syncthreads();
                    reduce(smem, blockDim.x);
                    if (threadIdx.x == 0)
                    {
                        pixel_jacobians_tR[global_pixel_idx * 6 * spline_deg_k + i] = smem[0] / float(num_vir_poses_per_frame);
                    }
                }
            }
        }

        /** 
         *  This kernel computes pixelwise hessian & gradient with repect to its corresponding 
         *  4 control knots. Patchwise reduction is then performed. For efficiency, we only compute 
         *  half of the hessian matrix (i.e., (12+1) * 12 / 6 elements) + the gradient vector + r**2 for each 
         *  patch. To launch this kernel, we need to have 2 dimensional blocks, and each block should 
         *  have PATCH_SIZE threads.
         */
        __global__ void kernel_compute_patch_cost_gradient_hessian(const int num_frames,
                                                                   const int num_keypoints,
                                                                   const int patch_size,
                                                                   const int spline_deg_k,
                                                                   const double *pixel_residuals,
                                                                   const double *pixel_jacobians,
                                                                   const double huber_a,
                                                                   const double inv_num_residuals,
                                                                   double *patch_cost_gradient_hessian)
        {
            assert(gridDim.x == num_frames);
            assert(gridDim.y == num_keypoints);
            assert(blockDim.x == patch_size);

            const int frameIdx = blockIdx.x;
            const int patchIdx = blockIdx.y;
            const int pixelIdx = threadIdx.x;

            const int global_pixel_idx = frameIdx * num_keypoints * patch_size + patchIdx * patch_size + pixelIdx;

            double row[25];
            int shift = global_pixel_idx * spline_deg_k * 6;
            row[0] = pixel_residuals[global_pixel_idx];
            // apply huber robust function
            const double huber_aa = huber_a * huber_a;
            const double x = 0.5 * row[0] * row[0];
            double sqrt_drho_dx = 1.;
            double rho = x;
            if (x > huber_aa)
            {
                sqrt_drho_dx = sqrtf(huber_a / (sqrtf(x) + 1e-8));
                rho = 2 * huber_a * sqrtf(x) - huber_aa;
            }
            //
            row[0] = sqrt_drho_dx * row[0];
            if (pixel_jacobians != nullptr)
            {
                for (int i = 0; i < 6 * spline_deg_k; i++)
                {
                    row[i + 1] = sqrt_drho_dx * pixel_jacobians[shift++];
                }
            }
            __syncthreads();

            const int ndim = spline_deg_k * 6 + 1;
            const int num_elems = (ndim + 1) * ndim / 2;
            const int offset = (frameIdx * num_keypoints + patchIdx) * num_elems;
            const int tid = threadIdx.x;
            extern __shared__ double smem[];
            if (pixel_jacobians != nullptr)
            {
                shift = 0;
                for (int i = 0; i < ndim; i++)
                {
                    for (int j = i; j < ndim; j++)
                    {
                        smem[tid] = row[i] * row[j];
                        __syncthreads();
                        reduce(smem, blockDim.x);
                        if (tid == 0)
                        {
                            patch_cost_gradient_hessian[offset + shift++] = smem[0] * inv_num_residuals;
                        }
                    }
                }
            }

            smem[tid] = rho;
            __syncthreads();
            reduce(smem, blockDim.x);
            if (tid == 0)
            {
                patch_cost_gradient_hessian[offset] = smem[0] * inv_num_residuals;
            }
        }

        /**
         *  This kernel reduces the patchwise costs, gradients, and hessians per frame; Each patch has 
         *  an associated vector with 91 elements. We accumulate the sum of all patches with respect to the 
         *  91 elements. The resulted "frame_cost_gradient_hessian" would have a dimension "N_FRAMES * 91". 
         *  To launch the kernel, we need to have "N_FRAMES * 91" blocks and each block has 256 threads;
         */
        __global__ void kernel_compute_frame_cost_gradient_hessian(const int num_frames,
                                                                   const int num_keypoints,
                                                                   const int spline_deg_k,
                                                                   const double *patch_cost_gradient_hessian,
                                                                   const unsigned char *keypoints_outlier_flags,
                                                                   double *frame_cost_gradient_hessian)
        {
            const int ndim = spline_deg_k * 6 + 1;
            const int num_elems = (ndim + 1) * ndim / 2;

            assert(gridDim.x == num_frames);
            assert(gridDim.y == num_elems || gridDim.y == 1);
            assert(blockDim.x == 256);

            const int frameIdx = blockIdx.x;
            const int patchEntryIdx = blockIdx.y;

            double sum = 0;
            for (int i = threadIdx.x; i < num_keypoints; i += 256)
            {
                if (keypoints_outlier_flags != nullptr && keypoints_outlier_flags[i] == 1)
                {
                    continue;
                }
                sum += *(patch_cost_gradient_hessian + (frameIdx * num_keypoints + i) * num_elems + patchEntryIdx);
            }

            __shared__ double smem[256];
            smem[threadIdx.x] = sum;
            __syncthreads();
            reduce(smem, 256);

            if (threadIdx.x == 0)
            {
                *(frame_cost_gradient_hessian + frameIdx * num_elems + patchEntryIdx) = smem[0];
            }
        }

        void compute_pixel_jacobian_residual(const unsigned char *I_ref,
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
                                             FLOAT *jacobian_pixel_to_ctrl_knots_tR,
                                             double *pixel_residuals,
                                             double *pixel_jacobians_tR)
        {
            const double fx = intrinsics.values[0];
            const double fy = intrinsics.values[1];
            const double cx = intrinsics.values[2];
            const double cy = intrinsics.values[3];
            kernel_compute_pixel_jacobian_residual<<<dim3(num_frames, num_keypoints, patch_size), dim3(num_vir_poses_per_frame), sizeof(double) * num_vir_poses_per_frame>>>(
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
                im_size_HW,
                jacobian_pixel_to_ctrl_knots_tR,
                pixel_residuals,
                pixel_jacobians_tR);
            cudaDeviceSynchronize();
        }

        void compute_patch_cost_gradient_hessian(const int num_frames,
                                                 const int num_keypoints,
                                                 const int patch_size,
                                                 const int spline_deg_k,
                                                 const double *pixel_residuals,
                                                 const double *pixel_jacobians,
                                                 const double huber_a,
                                                 const double inv_num_residuals,
                                                 double *patch_cost_gradient_hessian)
        {
            kernel_compute_patch_cost_gradient_hessian<<<dim3(num_frames, num_keypoints), dim3(patch_size), sizeof(double) * patch_size>>>(
                num_frames,
                num_keypoints,
                patch_size,
                spline_deg_k,
                pixel_residuals,
                pixel_jacobians,
                huber_a,
                inv_num_residuals,
                patch_cost_gradient_hessian);
            cudaDeviceSynchronize();
        }

        void compute_frame_cost_gradient_hessian(const int num_frames,
                                                 const int num_keypoints,
                                                 const int spline_deg_k,
                                                 const double *patch_cost_gradient_hessian,
                                                 const bool eval_gradient_hessian,
                                                 const unsigned char *keypoints_outlier_flags,
                                                 double *frame_cost_gradient_hessian)
        {
            if (eval_gradient_hessian)
            {
                const int ndim = spline_deg_k * 6 + 1;
                const int num_elems = (ndim + 1) * ndim / 2;
                kernel_compute_frame_cost_gradient_hessian<<<dim3(num_frames, num_elems), dim3(256)>>>(
                    num_frames,
                    num_keypoints,
                    spline_deg_k,
                    patch_cost_gradient_hessian,
                    keypoints_outlier_flags,
                    frame_cost_gradient_hessian);
            }
            else
            {
                kernel_compute_frame_cost_gradient_hessian<<<dim3(num_frames, 1), dim3(256)>>>(
                    num_frames,
                    num_keypoints,
                    spline_deg_k,
                    patch_cost_gradient_hessian,
                    keypoints_outlier_flags,
                    frame_cost_gradient_hessian);
            }
            cudaDeviceSynchronize();
        }
    } // namespace VO
} // namespace SLAM

#endif