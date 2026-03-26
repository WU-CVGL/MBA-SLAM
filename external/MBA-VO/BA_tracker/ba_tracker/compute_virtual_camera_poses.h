#ifndef SLAM_VO_SAMPLE_VIRTUAL_CAMERA_POSES_H
#define SLAM_VO_SAMPLE_VIRTUAL_CAMERA_POSES_H

#include "BA_tracker/core/common/CudaDefs.h"
#include "BA_tracker/core/common/SplineFunctor.h"
#include <cuda_runtime_api.h>

namespace SLAM
{
    namespace VO
    {
        /**
         *  This function computes the virtual camera poses during the exposure time for each frame.
         *  
         *  @param n_vir_poses_per_frame the number of virtual camera poses to be sampled for each frame;
         *  @param n_frames the number of image frames to be reblurred;
         */
        void compute_virtual_camera_poses(const int n_vir_poses_per_frame,
                                          const int n_frames,
                                          const double *img_cap_time,
                                          const double *img_exp_time,
                                          const int spline_deg_k,
                                          const double spline_start_time,
                                          const double spline_sample_interval,
                                          const double *spline_ctrl_knots_data_t,
                                          const double *spline_ctrl_knots_data_R,
                                          double *sampled_virtual_poses,
                                          double *jacobian_virtual_pose_t_to_ctrl_knots = nullptr,
                                          double *jacobian_virtual_pose_R_to_ctrl_knots = nullptr,
                                          double *jacobian_log_exp = nullptr,
                                          double *temp_X_4x4 = nullptr,
                                          double *temp_Y_4x4 = nullptr,
                                          double *temp_Z_4x4 = nullptr);
    } // namespace VO
} // namespace SLAM

#endif