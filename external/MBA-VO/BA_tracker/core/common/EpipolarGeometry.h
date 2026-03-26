#ifndef SLAM_CORE_COMMON_EPIPOLAR_GEOMETRY
#define SLAM_CORE_COMMON_EPIPOLAR_GEOMETRY

#include <Eigen/Dense>
#include "Linalg.h"

namespace SLAM
{
    namespace Core 
    {
        template<class T>
        Eigen::Matrix<T,3,3> essential_matrix(const Eigen::Matrix<T,3,3>& R, const Eigen::Matrix<T,3,1>& t)
        {
            return R * skew_matrix<T>(t);
        }

        template<class T>
        Eigen::Matrix<T,3,3> fundamental_matrix(const Eigen::Matrix<T,3,3>& Kinv_cur, 
                                                const Eigen::Matrix<T,4,4>& T_ref2cur, 
                                                const Eigen::Matrix<T,3,3>& Kinv_ref)
        {
            const Eigen::Matrix<T,3,3>& R = T_ref2cur.block(0,0,3,3);
            const Eigen::Matrix<T,3,1>& t = T_ref2cur.block(0,3,3,1);
            Eigen::Matrix<T,3,1> _t = -R.transpose() * t;
            return Kinv_cur.transpose() * essential_matrix<T>(R, _t) * Kinv_ref;
        }

        // The following two-view triangulation function was originally implemented 
        // in ColMap
        template<class T>
        Eigen::Matrix<T,3,1> TriangulatePoint(const Eigen::Matrix<T,3,4>& proj_matrix1,
                                              const Eigen::Matrix<T,3,4>& proj_matrix2,
                                              const Eigen::Matrix<T,2,1>& point1,
                                              const Eigen::Matrix<T,2,1>& point2) 
        {
            Eigen::Matrix<T,4,4> A;
            A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
            A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
            A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
            A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);
            Eigen::JacobiSVD<Eigen::Matrix<T,4,4>> svd(A, Eigen::ComputeFullV);
            return svd.matrixV().col(3).hnormalized();
        }
    }
}

#endif