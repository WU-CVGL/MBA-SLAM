//
// Created by peidong on 05.11.17.
//

#ifndef SLAM_CORE_DISTORTIONRADTAN_H
#define SLAM_CORE_DISTORTIONRADTAN_H

#include <Eigen/Dense>
#include <vector>

namespace SLAM
{
    namespace Core
    {
        class DistortionRadTan
        {
        public:
            DistortionRadTan(double k1, double k2, double p1, double p2);

        public:
            std::vector<double> getDistortionParams() const;
            void distort(const Eigen::Vector2d &point, Eigen::Vector2d &dPoint) const;
            void undistort(const Eigen::Vector2d &point, Eigen::Vector2d &uPoint) const;

        private:
            void distort(const Eigen::Vector2d &point,
                         Eigen::Vector2d &dPoint,
                         Eigen::Matrix2d &jacobian) const;

        private:
            double mk1, mk2, mp1, mp2;
        };
    } // namespace Core
} // namespace SLAM

#endif
