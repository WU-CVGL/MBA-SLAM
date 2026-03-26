#include "spline_update_step.h"
#include "BA_tracker/core/common/Time.h"
#include <iostream>

namespace SLAM
{
    namespace VO
    {
        void initialize_shared_cuda_storages(const int max_num_frames,
                                             const int max_num_virtual_poses_per_frame,
                                             const int max_num_keypoints,
                                             const int max_patch_size,
                                             const int max_num_ctrl_knots,
                                             const int spline_deg_k,
                                             CudaSharedStorages &storages)
        {
            const int num_virtual_poses = max_num_frames * max_num_virtual_poses_per_frame;
            const int num_patches = max_num_frames * max_num_keypoints;
            const int num_pixels = num_patches * max_patch_size;

            //
            cudaMalloc((void **)&storages.cuda_img_cap_time, sizeof(double) * max_num_frames);
            cudaMalloc((void **)&storages.cuda_img_exp_time, sizeof(double) * max_num_frames);
            cudaMalloc((void **)&storages.cuda_keypoint_depth_z, sizeof(double) * max_num_keypoints);
            cudaMalloc((void **)&storages.cuda_local_patch_pattern_xy, sizeof(int) * max_patch_size * 2);
            cudaMalloc((void **)&storages.cuda_cur_images, sizeof(void *) * max_num_frames);
            cudaMalloc((void **)&storages.cuda_keypoint_xy, sizeof(Core::Vector2d) * max_num_keypoints);
            cudaMalloc((void **)&storages.cuda_keypoints_outlier_flags, sizeof(unsigned char) * max_num_keypoints);

            cudaMalloc((void **)&storages.cuda_spline_ctrl_knots_data_t, sizeof(double) * max_num_ctrl_knots * 3);
            cudaMalloc((void **)&storages.cuda_spline_ctrl_knots_data_R, sizeof(double) * max_num_ctrl_knots * 4);

            //
            cudaMalloc((void **)&storages.cuda_sampled_virtual_poses, sizeof(double) * num_virtual_poses * 7);
            cudaMalloc((void **)&storages.cuda_J_virtual_pose_t_to_knots_t, sizeof(double) * num_virtual_poses * 3 * 3 * spline_deg_k);
            cudaMalloc((void **)&storages.cuda_J_virtual_pose_R_to_knots_R, sizeof(double) * num_virtual_poses * 4 * 3 * spline_deg_k);
            cudaMalloc((void **)&storages.cuda_jacobian_log_exp, sizeof(double) * num_virtual_poses * (spline_deg_k - 1) * 24);
            cudaMalloc((void **)&storages.cuda_temp_X_4x4, sizeof(double) * num_virtual_poses * 16);
            cudaMalloc((void **)&storages.cuda_temp_Y_4x4, sizeof(double) * num_virtual_poses * 16);
            cudaMalloc((void **)&storages.cuda_temp_Z_4x4, sizeof(double) * num_virtual_poses * 16);

            //
            cudaMalloc((void **)&storages.cuda_local_patches_XY, sizeof(Core::Vector2d) * num_patches);

            //
            cudaMalloc((void **)&storages.cuda_pixel_residuals, sizeof(double) * num_pixels);
            cudaMalloc((void **)&storages.cuda_pixel_jacobians_tR, sizeof(double) * num_pixels * spline_deg_k * 6);
            cudaMalloc((void **)&storages.cuda_vir_pixel_to_ctrl_knots_tR, sizeof(FLOAT) * num_pixels * spline_deg_k * 6 * max_num_virtual_poses_per_frame);
            cudaMalloc((void **)&storages.cuda_vir_pixel_residual, sizeof(FLOAT) * num_pixels * max_num_virtual_poses_per_frame);

            //
            int nelems = spline_deg_k * 6 + 1;
            nelems = (1 + nelems) * nelems / 2;

            cudaMalloc((void **)&storages.cuda_patch_cost_gradient_hessian_tR, sizeof(double) * num_patches * nelems);
            cudaMalloc((void **)&storages.cuda_frame_cost_gradient_hessian_tR, sizeof(double) * max_num_frames * nelems);
        }

        void free_shared_cuda_storages(CudaSharedStorages &storages)
        {
            cudaFree(storages.cuda_img_cap_time);
            cudaFree(storages.cuda_img_exp_time);
            cudaFree(storages.cuda_keypoint_depth_z);
            cudaFree(storages.cuda_local_patch_pattern_xy);
            cudaFree(storages.cuda_cur_images);
            cudaFree(storages.cuda_keypoint_xy);
            cudaFree(storages.cuda_keypoints_outlier_flags);

            cudaFree(storages.cuda_spline_ctrl_knots_data_t);
            cudaFree(storages.cuda_spline_ctrl_knots_data_R);

            //
            cudaFree(storages.cuda_sampled_virtual_poses);
            cudaFree(storages.cuda_J_virtual_pose_t_to_knots_t);
            cudaFree(storages.cuda_J_virtual_pose_R_to_knots_R);
            cudaFree(storages.cuda_jacobian_log_exp);
            cudaFree(storages.cuda_temp_X_4x4);
            cudaFree(storages.cuda_temp_Y_4x4);
            cudaFree(storages.cuda_temp_Z_4x4);

            //
            cudaFree(storages.cuda_local_patches_XY);

            //
            cudaFree(storages.cuda_vir_pixel_to_ctrl_knots_tR);
            cudaFree(storages.cuda_vir_pixel_residual);
            cudaFree(storages.cuda_pixel_residuals);
            cudaFree(storages.cuda_pixel_jacobians_tR);

            //
            cudaFree(storages.cuda_patch_cost_gradient_hessian_tR);

            //
            cudaFree(storages.cuda_frame_cost_gradient_hessian_tR);
        }

        void evaluate_cost_hessian_gradient(const int n_vir_poses_per_frame,
                                            const int n_frames,
                                            const unsigned char *cuda_ref_img,
                                            const float *cuda_dIxy_ref,
                                            const int num_keypoints,
                                            const int patch_size,
                                            const Core::VectorX<double, 4> &intrinsics,
                                            const Core::VectorX<int, 2> &im_size_HW,
                                            const int spline_deg_k,
                                            const double spline_start_time,
                                            const double spline_sample_dt,
                                            const int *cpu_ctrl_knot_start_indices,
                                            const int num_ctrl_knots,
                                            const CudaSharedStorages &storages,
                                            const double huber_a,
                                            double *total_costs,
                                            double *cpu_hessian_tR,
                                            double *cpu_gradient_tR)
        {
            const int num_residuals = (num_keypoints - storages.num_bad_keypoints) * n_frames * patch_size;
            const double inv_num_residuals = 1.0 / num_residuals;

            // TODO: it seems the optimized version is less stable than the original version. We need to
            //       figure out what the reason is!
            bool use_optimized = false;

            if (cpu_hessian_tR != nullptr)
            {
                // compute virtual camera poses & jacobians
                // Core::WallTime start_time = Core::currentTime();
                compute_virtual_camera_poses(n_vir_poses_per_frame,
                                             n_frames,
                                             storages.cuda_img_cap_time,
                                             storages.cuda_img_exp_time,
                                             spline_deg_k,
                                             spline_start_time,
                                             spline_sample_dt,
                                             storages.cuda_spline_ctrl_knots_data_t,
                                             storages.cuda_spline_ctrl_knots_data_R,
                                             storages.cuda_sampled_virtual_poses,
                                             storages.cuda_J_virtual_pose_t_to_knots_t,
                                             storages.cuda_J_virtual_pose_R_to_knots_R,
                                             storages.cuda_jacobian_log_exp,
                                             storages.cuda_temp_X_4x4,
                                             storages.cuda_temp_Y_4x4,
                                             storages.cuda_temp_Z_4x4);
                // printf("--compute_virtual_camera_poses: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));

                // compute warped patch center position
                // start_time = Core::currentTime();
                compute_local_patches_xy(n_vir_poses_per_frame,
                                         n_frames,
                                         storages.cuda_sampled_virtual_poses,
                                         storages.cuda_keypoint_xy,
                                         storages.cuda_keypoint_depth_z,
                                         num_keypoints,
                                         intrinsics,
                                         im_size_HW,
                                         storages.cuda_local_patches_XY);
                // printf("--compute_local_patches_xy: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));

                // start_time = Core::currentTime();
                if (use_optimized == false)
                {
                    compute_pixel_jacobian_residual(cuda_ref_img,
                                                    cuda_dIxy_ref,
                                                    storages.cuda_cur_images,
                                                    n_vir_poses_per_frame,
                                                    n_frames,
                                                    storages.cuda_sampled_virtual_poses,
                                                    spline_deg_k,
                                                    storages.cuda_J_virtual_pose_t_to_knots_t,
                                                    storages.cuda_J_virtual_pose_R_to_knots_R,
                                                    storages.cuda_local_patches_XY,
                                                    storages.cuda_keypoint_depth_z,
                                                    num_keypoints,
                                                    storages.cuda_local_patch_pattern_xy,
                                                    patch_size,
                                                    intrinsics,
                                                    im_size_HW,
                                                    storages.cuda_vir_pixel_to_ctrl_knots_tR,
                                                    storages.cuda_pixel_residuals,
                                                    storages.cuda_pixel_jacobians_tR);
                }
                else
                {
                    compute_pixel_jacobian_residual_fast(cuda_ref_img,
                                                         cuda_dIxy_ref,
                                                         storages.cuda_cur_images,
                                                         n_vir_poses_per_frame,
                                                         n_frames,
                                                         storages.cuda_sampled_virtual_poses,
                                                         spline_deg_k,
                                                         storages.cuda_J_virtual_pose_t_to_knots_t,
                                                         storages.cuda_J_virtual_pose_R_to_knots_R,
                                                         storages.cuda_local_patches_XY,
                                                         storages.cuda_keypoint_depth_z,
                                                         num_keypoints,
                                                         storages.cuda_local_patch_pattern_xy,
                                                         patch_size,
                                                         intrinsics,
                                                         im_size_HW,
                                                         storages.cuda_vir_pixel_to_ctrl_knots_tR,
                                                         storages.cuda_vir_pixel_residual,
                                                         storages.cuda_pixel_residuals,
                                                         storages.cuda_pixel_jacobians_tR);
                }
                // printf("--compute_pixel_jacobian_residual: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));

                // compute patch cost, gradient & hessian
                // start_time = Core::currentTime();
                compute_patch_cost_gradient_hessian(n_frames,
                                                    num_keypoints,
                                                    patch_size,
                                                    spline_deg_k,
                                                    storages.cuda_pixel_residuals,
                                                    storages.cuda_pixel_jacobians_tR,
                                                    huber_a,
                                                    inv_num_residuals,
                                                    storages.cuda_patch_cost_gradient_hessian_tR);
                // printf("--compute_patch_cost_gradient_hessian: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));

                // compute frame cost, gradient & hessian
                // start_time = Core::currentTime();
                compute_frame_cost_gradient_hessian(n_frames,
                                                    num_keypoints,
                                                    spline_deg_k,
                                                    storages.cuda_patch_cost_gradient_hessian_tR,
                                                    true,
                                                    storages.cuda_keypoints_outlier_flags,
                                                    storages.cuda_frame_cost_gradient_hessian_tR);
                // printf("--compute_frame_cost_gradient_hessian: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));

                // merge cost, gradient & hessian
                // start_time = Core::currentTime();
                merge_hessian_gradient_cost(n_frames,
                                            spline_deg_k,
                                            storages.cuda_frame_cost_gradient_hessian_tR,
                                            cpu_ctrl_knot_start_indices,
                                            num_ctrl_knots,
                                            total_costs,
                                            cpu_hessian_tR,
                                            cpu_gradient_tR);
                // printf("--merge_hessian_gradient_cost: %f us\n", Core::elapsedTimeInMicroSeconds(start_time));
            }
            else
            {
                // compute virtual camera poses & jacobians
                compute_virtual_camera_poses(n_vir_poses_per_frame,
                                             n_frames,
                                             storages.cuda_img_cap_time,
                                             storages.cuda_img_exp_time,
                                             spline_deg_k,
                                             spline_start_time,
                                             spline_sample_dt,
                                             storages.cuda_spline_ctrl_knots_data_t,
                                             storages.cuda_spline_ctrl_knots_data_R,
                                             storages.cuda_sampled_virtual_poses,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr);

                // compute warped patch center position
                compute_local_patches_xy(n_vir_poses_per_frame,
                                         n_frames,
                                         storages.cuda_sampled_virtual_poses,
                                         storages.cuda_keypoint_xy,
                                         storages.cuda_keypoint_depth_z,
                                         num_keypoints,
                                         intrinsics,
                                         im_size_HW,
                                         storages.cuda_local_patches_XY);

                if (use_optimized == false)
                {
                    compute_pixel_jacobian_residual(cuda_ref_img,
                                                    cuda_dIxy_ref,
                                                    storages.cuda_cur_images,
                                                    n_vir_poses_per_frame,
                                                    n_frames,
                                                    storages.cuda_sampled_virtual_poses,
                                                    spline_deg_k,
                                                    storages.cuda_J_virtual_pose_t_to_knots_t,
                                                    storages.cuda_J_virtual_pose_R_to_knots_R,
                                                    storages.cuda_local_patches_XY,
                                                    storages.cuda_keypoint_depth_z,
                                                    num_keypoints,
                                                    storages.cuda_local_patch_pattern_xy,
                                                    patch_size,
                                                    intrinsics,
                                                    im_size_HW,
                                                    nullptr,
                                                    storages.cuda_pixel_residuals,
                                                    nullptr);
                }
                else
                {
                    compute_pixel_jacobian_residual_fast(cuda_ref_img,
                                                         cuda_dIxy_ref,
                                                         storages.cuda_cur_images,
                                                         n_vir_poses_per_frame,
                                                         n_frames,
                                                         storages.cuda_sampled_virtual_poses,
                                                         spline_deg_k,
                                                         storages.cuda_J_virtual_pose_t_to_knots_t,
                                                         storages.cuda_J_virtual_pose_R_to_knots_R,
                                                         storages.cuda_local_patches_XY,
                                                         storages.cuda_keypoint_depth_z,
                                                         num_keypoints,
                                                         storages.cuda_local_patch_pattern_xy,
                                                         patch_size,
                                                         intrinsics,
                                                         im_size_HW,
                                                         nullptr,
                                                         storages.cuda_vir_pixel_residual,
                                                         storages.cuda_pixel_residuals,
                                                         nullptr);
                }

                // compute patch cost, gradient & hessian
                compute_patch_cost_gradient_hessian(n_frames,
                                                    num_keypoints,
                                                    patch_size,
                                                    spline_deg_k,
                                                    storages.cuda_pixel_residuals,
                                                    nullptr,
                                                    huber_a,
                                                    inv_num_residuals,
                                                    storages.cuda_patch_cost_gradient_hessian_tR);

                // compute frame cost, gradient & hessian
                compute_frame_cost_gradient_hessian(n_frames,
                                                    num_keypoints,
                                                    spline_deg_k,
                                                    storages.cuda_patch_cost_gradient_hessian_tR,
                                                    false,
                                                    storages.cuda_keypoints_outlier_flags,
                                                    storages.cuda_frame_cost_gradient_hessian_tR);

                // merge cost, gradient & hessian
                merge_hessian_gradient_cost(n_frames,
                                            spline_deg_k,
                                            storages.cuda_frame_cost_gradient_hessian_tR,
                                            cpu_ctrl_knot_start_indices,
                                            num_ctrl_knots,
                                            total_costs,
                                            nullptr,
                                            nullptr);
            }
        }
    } // namespace VO
} // namespace SLAM