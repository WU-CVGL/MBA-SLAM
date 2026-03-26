#ifndef SLAM_UTILS_SPLINE_TRAJECTORY
#define SLAM_UTILS_SPLINE_TRAJECTORY

/*------------------------------------------------------------------------------*
* '' Spline class for B-Spline curve of camera trajectory ``
* This file is modified based on rrd-slam from https://github.com/jaehak/rrd_slam 
* by Jae-Hak Kim, Cesar Cadena and Ian Reid
* Reference: "Spline Fusion: A continuous-time representation for visual - 
* inertial fusion with application to rolling shutter cameras"
*------------------------------------------------------------------------------*/

#include "BA_tracker/core/states/Transformation.h"
#include <Eigen/Dense>
#include <stdio.h>
#include <vector>

namespace SLAM
{
    namespace Utils
    {
        class SplineBase
        {
        public:
            SplineBase(double dt_between_control_knots,
                       int n_contrl_knots_per_segment,
                       double gravity,
                       Eigen::Vector3d bacc,
                       Eigen::Vector3d bgyro);

        public:
            void insert_control_knot(Core::Transformation &T_b2w);

        protected:
            std::vector<Core::Transformation> mControlKnots;
            double mdtBetwContrlKnots;
            int mnContrlKnotsPerSegment;
            double mGravity;
            Eigen::Vector3d mBiasAcc;
            Eigen::Vector3d mBiasGyro;
        };

        class SplineCubic : public SplineBase
        {
        public:
            SplineCubic(double dt_between_control_knots,
                        double gravity,
                        Eigen::Vector3d bacc,
                        Eigen::Vector3d bgyro);

        public:
            bool get_interpolation(double t, Core::Transformation &T_b2w);
            bool get_interpolation(double t,
                                   Core::Transformation &T_b2w,
                                   Eigen::Vector3d &velocity_w,
                                   Eigen::Vector3d &gyro_b,
                                   Eigen::Vector3d &acc_b);

        private:
            Eigen::Matrix4d mC4; // Basis for k = 4
        };
    } // namespace Utils
} // namespace SLAM

#endif
