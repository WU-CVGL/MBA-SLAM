//
// Created by peidong on 2/16/20.
//

#include "Geometry.h"

namespace SLAM
{
    namespace Utils
    {
        void convert_ray_d_to_z(Core::CameraBase::Ptr camPtr, Core::Image<float>::Ptr ray_d, Core::Image<float>::Ptr ray_z)
        {
            size_t H = ray_d->nHeight();
            size_t W = ray_d->nWidth();
            Eigen::Vector3d principle_axis(0, 0, 1);

            float *ray_d_ptr = ray_d->getData();
            float *ray_z_ptr = ray_z->getData();

            for (int r = 0; r < H; r++)
            {
                for (int c = 0; c < W; ++c, ++ray_d_ptr, ++ray_z_ptr)
                {
                    Eigen::Vector2d p2d(c, r);
                    Eigen::Vector3d P3d;
                    if (!camPtr->unproject(p2d, 1, P3d))
                    {
                        ray_z_ptr[0] = 0;
                        continue;
                    }

                    double cos_theta = P3d.dot(principle_axis) / P3d.norm();
                    ray_z_ptr[0] = ray_d_ptr[0] * cos_theta;
                }
            }
        }

        void convert_depth_z_to_pcl(Core::CameraBase::Ptr camPtr, Core::Image<float>::Ptr ray_z, Core::Image<float>::Ptr pcl)
        {
            size_t H = ray_z->nHeight();
            size_t W = ray_z->nWidth();
            float *ray_z_ptr = ray_z->getData();
            float *pcl_ptr = pcl->getData();

            for (int r = 0; r < H; ++r)
            {
                for (int c = 0; c < W; ++c, ++ray_z_ptr, pcl_ptr += 3)
                {
                    Eigen::Vector2d p2d(c, r);
                    Eigen::Vector3d P3d;
                    if (!camPtr->unproject(p2d, ray_z_ptr[0], P3d))
                    {
                        pcl_ptr[0] = 0;
                        pcl_ptr[1] = 0;
                        pcl_ptr[2] = 0;
                    }
                    pcl_ptr[0] = P3d(0);
                    pcl_ptr[1] = P3d(1);
                    pcl_ptr[2] = P3d(2);
                }
            }
        }

        double point_to_line_distance(Eigen::Vector2d point, Eigen::Vector3d line)
        {
            double a = line(0);
            double b = line(1);
            double c = line(2);

            double x = point(0);
            double y = point(1);

            return fabs(a * x + b * y + c) / sqrt(a * a + b * b);
        }
    } // namespace Utils
} // namespace SLAM