//
// Created by peidong on 2/13/20.
//

#ifndef SLAM_CAMERAUNIFIED_H
#define SLAM_CAMERAUNIFIED_H

#include "CameraBase.h"

namespace SLAM
{
    namespace Core
    {
        class CameraUnified : public CameraBase
        {

        public:
            CameraUnified(const Eigen::Vector4d& K, double xi, int H, int W);
            CameraUnified(const Transformation& T_body_to_sensor,
                          const Eigen::Vector4d &K, double xi, int H, int W);

            bool project(const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d);
            bool project(const int pyra_level, const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d);
            bool unproject(const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d);
            bool unproject(const int pyra_level, const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d);

            void projection_jacobian(Eigen::Vector3d &P3d,
                                     Eigen::Matrix<double, 2, 3> &jacobian);

            std::vector<double> getCameraParams();

            template <typename T>
            static Eigen::Matrix<T, 2, 1> project(std::vector<T> camera_params, Eigen::Matrix<T, 3, 1> P3d);

        private:
            double m_xi;
        };

        template <typename T>
        Eigen::Matrix<T, 2, 1> CameraUnified::project(std::vector<T> camera_params, Eigen::Matrix<T, 3, 1> P3d)
        {
            T d = P3d.norm();
            T rz = 1.0 / (P3d(2) + camera_params.at(4) * d);

            // Project the scene point to the normalized plane.
            Eigen::Matrix<T, 2, 1> normalizedPoint(P3d(0) * rz, P3d(1) * rz);

            T x = camera_params.at(0) * normalizedPoint(0) + camera_params.at(2);
            T y = camera_params.at(1) * normalizedPoint(1) + camera_params.at(3);

            return Eigen::Matrix<T, 2, 1>(x, y);
        }
    } // namespace Core
} // namespace SLAM

#endif //SLAM_CAMERAPINHOLE_H
