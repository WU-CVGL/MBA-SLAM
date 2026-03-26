#ifndef SLAM_VO_COMPUTE_LOCAL_PATCHES_H
#define SLAM_VO_COMPUTE_LOCAL_PATCHES_H

#include "BA_tracker/core/common/Vector.h"

namespace SLAM
{
    namespace VO
    {
        void compute_local_patches_xy(const int num_virtual_poses_per_frame,
                                      const int num_frames,
                                      const double *virtual_cam_poses,
                                      const Core::Vector2d *sparse_keypoints,
                                      const double *sparse_keypoints_z,
                                      const int num_keypoints,
                                      const Core::VectorX<double, 4> &intrinsics,
                                      const Core::VectorX<int, 2> &im_HW,
                                      Core::Vector2d *local_patches_xy);
    } // namespace VO
} // namespace SLAM

#endif