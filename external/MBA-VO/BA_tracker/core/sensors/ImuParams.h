#ifndef SLAM_CORE_SENSORS_IMUPARAMS
#define SLAM_CORE_SENSORS_IMUPARAMS

#include <Eigen/Dense>

namespace SLAM
{
    namespace Core
    {
        struct ImuParams
        {
            double accelerometerDriftNoiseDensity = 0.0;
            double accelerometerNoiseDensity = 0.0;
            double accelerometerSaturation = 0.0;
            double accelerometerScaleFactor = 1.0;

            double gyroscopeDriftNoiseDensity = 0.0;
            double gyroscopeNoiseDensity = 0.0;
            double gyroscopeSaturation = 0.0;
            double gyroscopeScaleFactor = 1.0;

            Eigen::Vector3d gravityVector = Eigen::Vector3d(0, 0, -9.781);
        };
    } // namespace Core
} // namespace SLAM

#endif