//
// Created by peidong on 2/13/20.
//

#ifndef SLAM_CAMERAPINHOLE_H
#define SLAM_CAMERAPINHOLE_H

#include <memory>

#include "CameraBase.h"


namespace SLAM
{
    namespace Core
    {
        class CameraPinhole : public CameraBase
        {
        public:
            using Ptr = std::shared_ptr<CameraPinhole>;
            CameraPinhole(const Eigen::Vector4d &K, int H, int W);
            CameraPinhole(const Transformation &T_body_to_sensor,
                          const Eigen::Vector4d& K, int H, int W);

            bool project(const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d);
            bool project(const int pyra_level, const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d);
            bool unproject(const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d);
            bool unproject(const int pyra_level, const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d);

            void projection_jacobian(Eigen::Vector3d &P3d,
                                     Eigen::Matrix<double, 2, 3> &jacobian);

            std::vector<double> getCameraParams();

            template <typename T>
            static Eigen::Matrix<T, 2, 1> project(std::vector<T> camera_params, Eigen::Matrix<T, 3, 1> P3d);
        };

        template <typename T>
        Eigen::Matrix<T, 2, 1> CameraPinhole::project(std::vector<T> camera_params, Eigen::Matrix<T, 3, 1> P3d)
        {
            T xn = P3d(0) / (P3d(2) + 1e-8);
            T yn = P3d(1) / (P3d(2) + 1e-8);

            T x = camera_params.at(0) * xn + camera_params.at(2);
            T y = camera_params.at(1) * yn + camera_params.at(3);
            return Eigen::Matrix<T, 2, 1>(x, y);
        }
    } // namespace Core
} // namespace SLAM

#endif //SLAM_CAMERAPINHOLE_H
