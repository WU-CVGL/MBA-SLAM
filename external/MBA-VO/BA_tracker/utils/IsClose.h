#ifndef SRC_UTILS_ISCLOSE_H
#define SRC_UTILS_ISCLOSE_H

#include <cmath>
#include <limits>

namespace SLAM
{
    namespace Utils
    {
        template <typename T>
        bool is_close(T a, T b, T epsilon = std::numeric_limits<T>::epsilon())
        {
            return std::fabs(a-b) < epsilon;
        }
    }
}

#endif