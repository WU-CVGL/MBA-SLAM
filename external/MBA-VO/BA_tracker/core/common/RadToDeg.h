#ifndef SLAM_CORE_COMMON_RAD_TO_DEG_H_
#define SLAM_CORE_COMMON_RAD_TO_DEG_H_

#define PI 3.1415926535897932384626433832795

namespace SLAM
{
    namespace Core
    {
        inline double rad_to_deg(double radians)
        {
            return radians * 180. / PI;
        }

        inline double deg_to_rad(double degrees)
        {
            return degrees * PI / 180.;
        }
    } // namespace Core
} // namespace SLAM

#endif