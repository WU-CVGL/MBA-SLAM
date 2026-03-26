#ifndef __SRC_CORE_IMU_H_
#define __SRC_CORE_IMU_H_

#include "SensorBase.h"
#include "ImuParams.h"

namespace SLAM
{
    namespace Core
    {
        class IMU : public SensorBase
        {
        public:
            IMU(ImuParams& params);
            
            const ImuParams& getParams();
        
        private:
            ImuParams mImuParams;
        }; 
    } 
} 

#endif
