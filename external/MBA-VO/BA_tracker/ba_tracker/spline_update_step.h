#ifndef SLAM_VO_SPLINE_UPDATE_STEP_H
#define SLAM_VO_SPLINE_UPDATE_STEP_H

#include "compute_hessian_gradients_cost.h"
#include "compute_local_patches_xy.h"
#include "compute_virtual_camera_poses.h"
#include "BA_tracker/core/common/CudaDefs.h"
#include "BA_tracker/core/common/CustomType.h"
#include "BA_tracker/core/common/Vector.h"
#include "merge_hessian_gradient_cost.h"
#include "solve_normal_equation.h"
#include <vector>

namespace SLAM
{
    namespace VO
    {
        struct CudaSharedStorages
        {
            // internal cuda memory
            double *cuda_img_cap_time = nullptr;
            double *cuda_img_exp_time = nullptr;
            double *cuda_keypoint_depth_z = nullptr;
            Core::Vector2d *cuda_keypoint_xy = nullptr;
            unsigned char *cuda_keypoints_outlier_flags = nullptr;
            int num_bad_keypoints = 0;

            unsigned char **cuda_cur_images = nullptr;
            int *cuda_local_patch_pattern_xy = nullptr;

            // cuda spline
            double *cuda_spline_ctrl_knots_data_t = nullptr;
            double *cuda_spline_ctrl_knots_data_R = nullptr;

            // internal cuda memory used to store sampled virtual poses & jacobians
            double *cuda_sampled_virtual_poses = nullptr;
            double *cuda_J_virtual_pose_t_to_knots_t = nullptr;
            double *cuda_J_virtual_pose_R_to_knots_R = nullptr;
            double *cuda_jacobian_log_exp = nullptr;
            double *cuda_temp_X_4x4 = nullptr;
            double *cuda_temp_Y_4x4 = nullptr;
            double *cuda_temp_Z_4x4 = nullptr;

            // internal cuda memory used to store patch patterns & local patch center positions
            Core::Vector2d *cuda_local_patches_XY = nullptr;

            // internal cuda memory used to store per pixel residuals & jacobians
            double *cuda_pixel_residuals = nullptr;
            double *cuda_pixel_jacobians_tR = nullptr;
            FLOAT *cuda_vir_pixel_to_ctrl_knots_tR = nullptr;
            FLOAT *cuda_vir_pixel_residual = nullptr;

            // internal cuda memory used to store per patch cost, gradient & jacobians
            double *cuda_patch_cost_gradient_hessian_tR = nullptr;

            // internal cuda memory used to store per frame cost, gradient & jacobians
            double *cuda_frame_cost_gradient_hessian_tR = nullptr;
        };

        void initialize_shared_cuda_storages(const int max_num_frames,
                                             const int max_num_virtual_poses_per_frame,
                                             const int max_num_keypoints,
                                             const int max_patch_size,
                                             const int max_num_ctrl_knots,
                                             const int spline_deg_k,
                                             CudaSharedStorages &storages);

        void free_shared_cuda_storages(CudaSharedStorages &storages);

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
                                            double *cpu_gradient_tR);
    } // namespace VO
} // namespace SLAM

#endif