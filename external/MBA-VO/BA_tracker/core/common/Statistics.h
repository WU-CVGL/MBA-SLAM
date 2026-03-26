#ifndef SLAM_CORE_COMMON_STATISTICS
#define SLAM_CORE_COMMON_STATISTICS

#include <cmath>
#include <vector>

namespace SLAM
{
    namespace Core
    {
        template <typename T>
        void mean_sigma(std::vector<T> &data, T &mean, T &sigma)
        {
            // compute mean
            mean = 0;
            for (size_t i = 0; i < data.size(); i++)
            {
                mean += data.at(i);
            }
            mean = mean / float(data.size());

            // compute standard derivation
            T var = 0;
            for (size_t i = 0; i < data.size(); i++)
            {
                var = var + (data.at(i) - mean) * (data.at(i) - mean);
            }
            var /= float(data.size());

            sigma = sqrt(var);
        }
    } // namespace Core
} // namespace SLAM

#endif