//
// Created by peidong on 2/16/20.
//

#ifndef SLAM_GEOMETRY_H
#define SLAM_GEOMETRY_H

#include "BA_tracker/core/sensors/CameraBase.h"
#include "BA_tracker/core/measurements/Image.h"

namespace SLAM
{
    namespace Utils {
        void convert_ray_d_to_z(Core::CameraBase::Ptr camPtr, Core::Image<float>::Ptr ray_d, Core::Image<float>::Ptr ray_z);

        void convert_depth_z_to_pcl(Core::CameraBase::Ptr camPtr, Core::Image<float>::Ptr ray_z, Core::Image<float>::Ptr pcl);

        double point_to_line_distance(Eigen::Vector2d point, Eigen::Vector3d line);
    } // namespace Utils
}

#endif //SLAM_GEOMETRY_H
