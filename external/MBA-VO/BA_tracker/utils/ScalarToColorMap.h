#ifndef SLAM_UTILS_SCALAR_TO_COLOR_MAP
#define SLAM_UTILS_SCALAR_TO_COLOR_MAP

#include "ColorMapJet.h"
#include <Eigen/Dense>
#include <algorithm>

namespace SLAM
{
    namespace Utils
    {
        inline Eigen::Vector3d ScalarToColorMap(double scalar, double min, double max)
        {
            double normalized = (scalar - min) / (max - min);
            int idx = (int)round(std::max(0.0, std::min(normalized, 1.0) * 255.0));

            Eigen::Vector3d color;
            color[0] = (unsigned char)floor(ColorMapJet[idx][2] * 255.0f + 0.5f);
            color[1] = (unsigned char)floor(ColorMapJet[idx][1] * 255.0f + 0.5f);
            color[2] = (unsigned char)floor(ColorMapJet[idx][0] * 255.0f + 0.5f);
            return color;
        }
    } // namespace Utils
} // namespace SLAM

#endif