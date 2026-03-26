#include "Imu.h"

namespace SLAM
{
    namespace Core
    {
        IMU::IMU(ImuParams& params)
        : SensorBase()
        , mImuParams(params)
        {}
        
        const ImuParams& IMU::getParams()
        {
            return mImuParams;
        }

    } 
} 
