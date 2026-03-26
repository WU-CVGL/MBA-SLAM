#ifndef SLAM_CORE_TIME_H_
#define SLAM_CORE_TIME_H_

#include <chrono>

namespace SLAM
{
    namespace Core
    {
        typedef std::chrono::high_resolution_clock Clock;
        typedef std::chrono::time_point<Clock> WallTime;

        inline WallTime currentTime()
        {
            return Clock::now();
        }

        inline double elapsedTimeInSeconds(const WallTime &start)
        {
            WallTime end = Clock::now();
            auto diff = end - start;
            return std::chrono::duration_cast<std::chrono::seconds>(diff).count();
        }

        inline double elapsedTimeInMilliSeconds(const WallTime &start)
        {
            WallTime end = Clock::now();
            auto diff = end - start;
            return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
        }

        inline double elapsedTimeInMicroSeconds(const WallTime &start)
        {
            WallTime end = Clock::now();
            auto diff = end - start;
            return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
        }
    } // namespace Core
} // namespace SLAM

#endif