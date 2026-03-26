#include "compute_virtual_camera_poses.h"
#include <assert.h>
#include <cstdio>

namespace SLAM
{
    namespace VO
    {
        __global__ void
        kernel_compute_virtual_camera_poses(const int n_vir_poses_per_frame,
                                            const double *img_cap_time,
                                            const double *img_exp_time,
                                            const int spline_deg_k,
                                            const double spline_start_time,
                                            const double spline_sample_interval,
                                            const double *spline_ctrl_knots_data_t,
                                            const double *spline_ctrl_knots_data_R,
                                            double *sampled_virtual_poses,
                                            double *jacobian_virtual_pose_t_to_ctrl_knots,
                                            double *jacobian_virtual_pose_R_to_ctrl_knots,
                                            double *jacobian_log_exp,
                                            double *temp_X_4x4,
                                            double *temp_Y_4x4,
                                            double *temp_Z_4x4)
        {
            assert(blockDim.x == n_vir_poses_per_frame);
            const int frame_idx = blockIdx.x;
            const int local_virtual_poses_idx = threadIdx.x;
            const int global_virtual_poses_idx = blockIdx.x * blockDim.x + threadIdx.x;

            const double t_cap = img_cap_time[frame_idx];
            const double t_mu = img_exp_time[frame_idx];
            const double t = t_cap - t_mu * 0.5 + local_virtual_poses_idx * t_mu / (n_vir_poses_per_frame - 1 + 1e-8);

            // compute the ctrl knots segment and local normalized time u
            int ctrl_knot_start_idx;
            double u;
            Core::SplineSegmentStartKnotIdxAndNormalizedU(t,
                                                          spline_start_time,
                                                          spline_sample_interval,
                                                          ctrl_knot_start_idx,
                                                          u);

            // interpolate camera poses
            double *J_t_to_ctrl_knot_t = nullptr;
            double *J_R_to_ctrl_knot_R = nullptr;
            double *J_log_exp = nullptr;
            double *J_tempX = nullptr;
            double *J_tempY = nullptr;
            double *J_tempZ = nullptr;
            if (jacobian_virtual_pose_t_to_ctrl_knots != nullptr)
            {
                J_t_to_ctrl_knot_t = jacobian_virtual_pose_t_to_ctrl_knots + global_virtual_poses_idx * 3 * 3 * spline_deg_k;
                J_R_to_ctrl_knot_R = jacobian_virtual_pose_R_to_ctrl_knots + global_virtual_poses_idx * 4 * 3 * spline_deg_k;
                J_log_exp = jacobian_log_exp + global_virtual_poses_idx * (spline_deg_k - 1) * 24;
                J_tempX = temp_X_4x4 + global_virtual_poses_idx * 16;
                J_tempY = temp_Y_4x4 + global_virtual_poses_idx * 16;
                J_tempZ = temp_Z_4x4 + global_virtual_poses_idx * 16;
            }

            Core::Vector3d t_c2r;
            Core::Quaterniond R_c2r;

            switch (spline_deg_k)
            {
            case 2:
            {
                t_c2r = Core::C2SplineVec3Functor(spline_ctrl_knots_data_t + ctrl_knot_start_idx * 3,
                                                  u,
                                                  J_t_to_ctrl_knot_t);

                R_c2r = Core::C2SplineRot3Functor(spline_ctrl_knots_data_R + ctrl_knot_start_idx * 4,
                                                  u,
                                                  J_R_to_ctrl_knot_R,
                                                  J_log_exp,
                                                  J_tempX,
                                                  J_tempY,
                                                  J_tempZ);

                break;
            }
            case 4:
            {
                t_c2r = Core::C4SplineVec3Functor(spline_ctrl_knots_data_t + ctrl_knot_start_idx * 3,
                                                  u,
                                                  J_t_to_ctrl_knot_t);

                R_c2r = Core::C4SplineRot3Functor(spline_ctrl_knots_data_R + ctrl_knot_start_idx * 4,
                                                  u,
                                                  J_R_to_ctrl_knot_R,
                                                  J_log_exp,
                                                  J_tempX,
                                                  J_tempY,
                                                  J_tempZ);

                break;
            }
            default:
                assert(false);
            }

            double *virtual_pose = sampled_virtual_poses + global_virtual_poses_idx * 7;
            virtual_pose[0] = t_c2r(0);
            virtual_pose[1] = t_c2r(1);
            virtual_pose[2] = t_c2r(2);
            virtual_pose[3] = R_c2r.x;
            virtual_pose[4] = R_c2r.y;
            virtual_pose[5] = R_c2r.z;
            virtual_pose[6] = R_c2r.w;
        }

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
                                          double *jacobian_virtual_pose_t_to_ctrl_knots,
                                          double *jacobian_virtual_pose_R_to_ctrl_knots,
                                          double *jacobian_log_exp,
                                          double *temp_X_4x4,
                                          double *temp_Y_4x4,
                                          double *temp_Z_4x4)
        {
            dim3 gridDim = dim3(n_frames);
            dim3 blockDim = dim3(n_vir_poses_per_frame);

            kernel_compute_virtual_camera_poses<<<gridDim, blockDim>>>(n_vir_poses_per_frame,
                                                                       img_cap_time,
                                                                       img_exp_time,
                                                                       spline_deg_k,
                                                                       spline_start_time,
                                                                       spline_sample_interval,
                                                                       spline_ctrl_knots_data_t,
                                                                       spline_ctrl_knots_data_R,
                                                                       sampled_virtual_poses,
                                                                       jacobian_virtual_pose_t_to_ctrl_knots,
                                                                       jacobian_virtual_pose_R_to_ctrl_knots,
                                                                       jacobian_log_exp,
                                                                       temp_X_4x4,
                                                                       temp_Y_4x4,
                                                                       temp_Z_4x4);

            cudaDeviceSynchronize();
        }
    } // namespace VO
} // namespace SLAM