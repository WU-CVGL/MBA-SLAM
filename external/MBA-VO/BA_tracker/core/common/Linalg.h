#ifndef SLAM_CORE_COMMON_LINALG
#define SLAM_CORE_COMMON_LINALG

#include <Eigen/Dense>

namespace SLAM
{
    namespace Core 
    {
        template<class T>
        Eigen::Matrix<T,3,3> skew_matrix(const Eigen::Matrix<T,3,1>& a)
        {
            Eigen::Matrix<T,3,3> A;
            A.setZero();
            
            A(0,1) = -a(2); A(0,2) = a(1);
            A(1,0) = a(2);  A(1,2) = -a(0);
            A(2,0) = -a(1); A(2,1) = a(0);

            return A;
        }
    }
}

#endif 