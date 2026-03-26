#ifndef __SRC_CORE_IMUMEASUREMENT_H_
#define __SRC_CORE_IMUMEASUREMENT_H_

#include <Eigen/Dense>

namespace SLAM
{
    namespace Core 
    {
        struct ImuMeasurement
        {
            double dt; // in second
            double timestamp; // in second
            Eigen::Vector3d acc;
            Eigen::Vector3d gyro;
        };
    }
}

#endif