#ifndef SLAM_CORE_COMMON_MATCHES_H
#define SLAM_CORE_COMMON_MATCHES_H

#include <Eigen/Dense>

namespace SLAM
{
    namespace Core
    {
        struct Match3D2D
        {
            int P3dId;               // 3D point Id
            int cameraId;            // keypoint camera Id
            Eigen::Vector2d p2d;     // keypoint position
            Eigen::Vector2d ref_p2d; // keypoint position in the latest keyframe
        };
    } // namespace Core
} // namespace SLAM
#endif