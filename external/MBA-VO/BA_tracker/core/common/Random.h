#ifndef SLAM_CORE_RANDOM_H_
#define SLAM_CORE_RANDOM_H_

#include <random>
#include <time.h>

namespace SLAM
{
    namespace Core
    {
        inline int random_int(int min, int max)
        {
            return rand() % (max - min) + min;
        }

        inline float random_float(float min, float max)
        {
            return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
        }
    } // namespace Core
} // namespace SLAM

#endif