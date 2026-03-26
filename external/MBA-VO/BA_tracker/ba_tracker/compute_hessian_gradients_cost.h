#ifndef SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H
#define SLAM_VO_COMPUTE_HESSIAN_GRADIENT_COST_H

#include "BA_tracker/core/common/CustomType.h"
#include "BA_tracker/core/common/Vector.h"

namespace SLAM
{
    namespace VO
    {
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
                                             double *pixel_jacobians_tR = nullptr);

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
                                                  FLOAT *jacobian_pixel_to_ctrl_knots_tR,
                                                  FLOAT *jacobian_vir_pixel_residuals,
                                                  double *pixel_residuals,
                                                  double *pixel_jacobians_tR = nullptr);

        void compute_patch_cost_gradient_hessian(const int num_frames,
                                                 const int num_keypoints,
                                                 const int patch_size,
                                                 const int spline_deg_k,
                                                 const double *pixel_residuals,
                                                 const double *pixel_jacobians,
                                                 const double huber_a,
                                                 const double inv_num_residuals,
                                                 double *patch_cost_gradient_hessian);

        void compute_frame_cost_gradient_hessian(const int num_frames,
                                                 const int num_keypoints,
                                                 const int spline_deg_k,
                                                 const double *patch_cost_gradient_hessian,
                                                 const bool eval_gradient_hessian,
                                                 const unsigned char *keypoints_outlier_flags,
                                                 double *frame_cost_gradient_hessian);

    } // namespace VO
} // namespace SLAM

#endif