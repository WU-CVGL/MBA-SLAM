#include "BA_tracker/core/common/Quaternion.h"
#include "BA_tracker/core/common/Vector.h"
#include <cuda_runtime_api.h>

namespace SLAM
{
    namespace VO
    {
        __global__ void kernel_compute_local_patches_xy(const int num_virtual_poses_per_frame,
                                                        const int num_frames,
                                                        const double *virtual_cam_poses,
                                                        const Core::Vector2d *sparse_keypoints,
                                                        const double *sparse_keypoints_z,
                                                        const int num_keypoints,
                                                        const Core::VectorX<double, 4> intrinsics,
                                                        const Core::VectorX<int, 2> im_HW,
                                                        Core::Vector2d *local_patches_xy)
        {
            const int global_local_patch_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (global_local_patch_idx >= num_frames * num_keypoints)
            {
                return;
            }
            const int frame_idx = global_local_patch_idx / num_keypoints;
            const int keypoint_idx = global_local_patch_idx % num_keypoints;
            const int pose_idx = frame_idx * num_virtual_poses_per_frame + num_virtual_poses_per_frame / 2;

            // back project keypoint
            const Core::Vector2d keypt_xy = sparse_keypoints[keypoint_idx];
            const double keypt_z = sparse_keypoints_z[keypoint_idx];

            Core::Vector3d P3dr;
            P3dr(0) = keypt_z * (keypt_xy(0) - intrinsics.values[2]) / intrinsics.values[0];
            P3dr(1) = keypt_z * (keypt_xy(1) - intrinsics.values[3]) / intrinsics.values[1];
            P3dr(2) = keypt_z;

            // compute 3D point in current frame
            const double *cam_pose = virtual_cam_poses + pose_idx * 7;
            const Core::Vector3d t_c2r(cam_pose[0], cam_pose[1], cam_pose[2]);
            const Core::Quaterniond R_c2r(cam_pose[3], cam_pose[4], cam_pose[5], cam_pose[6]);
            const Core::Quaterniond R_r2c = R_c2r.conjugate();
            const Core::Vector3d t_r2c = -(R_r2c * t_c2r);
            const Core::Vector3d P3dc = R_r2c * P3dr + t_r2c;

            // TODO: should we check if the point is valid here?
            // project into current image
            Core::Vector2d &local_patch_xy = local_patches_xy[global_local_patch_idx];
            local_patch_xy(0) = P3dc(0) / P3dc(2) * intrinsics.values[0] + intrinsics.values[2];
            local_patch_xy(1) = P3dc(1) / P3dc(2) * intrinsics.values[1] + intrinsics.values[3];
        }

        void compute_local_patches_xy(const int num_virtual_poses_per_frame,
                                      const int num_frames,
                                      const double *virtual_cam_poses,
                                      const Core::Vector2d *sparse_keypoints,
                                      const double *sparse_keypoints_z,
                                      const int num_keypoints,
                                      const Core::VectorX<double, 4> &intrinsics,
                                      const Core::VectorX<int, 2> &im_HW,
                                      Core::Vector2d *local_patches_xy)
        {
            int nThreads = num_frames * num_keypoints;
            dim3 gridDim = dim3(nThreads / 256 + 1);
            dim3 blockDim = dim3(256);

            kernel_compute_local_patches_xy<<<gridDim, blockDim>>>(num_virtual_poses_per_frame,
                                                                   num_frames,
                                                                   virtual_cam_poses,
                                                                   sparse_keypoints,
                                                                   sparse_keypoints_z,
                                                                   num_keypoints,
                                                                   intrinsics,
                                                                   im_HW,
                                                                   local_patches_xy);

            cudaDeviceSynchronize();
        }
    } // namespace VO
} // namespace SLAM
