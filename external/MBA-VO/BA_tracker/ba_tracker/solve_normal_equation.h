#ifndef SLAM_VO_SOLVE_NORMAL_EQUATION_H
#define SLAM_VO_SOLVE_NORMAL_EQUATION_H

#include <Eigen/Dense>

namespace SLAM
{
    namespace VO
    {
        template <typename DerivedA, typename Derivedb>
        void solve_normal_equation(const Eigen::MatrixBase<DerivedA> &A,
                                   const Eigen::MatrixBase<Derivedb> &b,
                                   const int SolverType,
                                   Eigen::MatrixBase<Derivedb> &x)
        {
            switch (SolverType)
            {
            case 0:
            {
                Eigen::JacobiSVD<DerivedA> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU);
                x = svd.solve(b);
                break;
            }
            case 1:
            {
                x = A.ldlt().solve(b);
                break;
            }
            default:
                assert(false && "Solver is not implemented...");
            }

            //
            x = -x;
        }
    } // namespace VO
} // namespace SLAM

#endif