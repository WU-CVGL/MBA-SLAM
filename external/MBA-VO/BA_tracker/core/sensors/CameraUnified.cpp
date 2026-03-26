//
// Created by peidong on 2/13/20.
//

#include "CameraUnified.h"

namespace SLAM
{
    namespace Core
    {
        CameraUnified::CameraUnified(const Eigen::Vector4d &K, double xi, int H, int W)
            : CameraBase(K, H, W), m_xi(xi)
        {
            mType = CAMERA_UNIFIED;
        }

        CameraUnified::CameraUnified(const Transformation &T_body_to_sensor,
                                     const Eigen::Vector4d& K, double xi, int H, int W)
            : CameraBase(T_body_to_sensor, K, H, W), m_xi(xi)
        {
            mType = CAMERA_UNIFIED;
        }

        bool CameraUnified::project(const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d)
        {
            if (P3d(2) < 0)
                return false;

            double d = P3d.norm();
            double rz = 1.0 / (P3d(2) + m_xi * d);

            // Project the scene point to the normalized plane.
            Eigen::Vector2d p2dn(P3d(0) * rz, P3d(1) * rz);

            if (mDistortion != nullptr)
            {
                mDistortion->distort(p2dn, p2dn);
            }

            p2d(0) = mK(0) * p2dn(0) + mK(2);
            p2d(1) = mK(1) * p2dn(1) + mK(3);

            return true;
        }

        bool CameraUnified::project(const int pyra_level, const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d)
        {
            if (P3d(2) < 0)
                return false;

            double d = P3d.norm();
            double rz = 1.0 / (P3d(2) + m_xi * d);

            // Project the scene point to the normalized plane.
            Eigen::Vector2d p2dn(P3d(0) * rz, P3d(1) * rz);

            if (mDistortion != nullptr)
            {
                mDistortion->distort(p2dn, p2dn);
            }

            double scale = 1. / pow(2, pyra_level);
            p2d(0) = scale * (mK(0) * p2dn(0) + mK(2));
            p2d(1) = scale * (mK(1) * p2dn(1) + mK(3));

            return true;
        }

        bool CameraUnified::unproject(const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d)
        {
            // normalize the pixel point with K^(-1)
            Eigen::Vector2d p2dn;
            p2dn(0) = (p2d(0) - mK(2)) / mK(0);
            p2dn(1) = (p2d(1) - mK(3)) / mK(1);

            if (mDistortion != nullptr)
            {
                mDistortion->undistort(p2dn, p2dn);
            }

            // Compute the unit ray vector that passes through the scene point.
            float rho2_u = p2dn.squaredNorm();
            float beta = 1.0 + (1.0 - m_xi * m_xi) * rho2_u;
            float lambda = (m_xi + sqrt(beta)) / (1.0 + rho2_u);

            if (beta < 0)
            {
                return false;
            }

            P3d(0) = lambda * p2dn(0);
            P3d(1) = lambda * p2dn(1);
            P3d(2) = lambda - m_xi;

            if (P3d(2) < 0)
            {
                return false;
            }

            P3d /= P3d(2);
            P3d *= z;

            return true;
        }

        bool CameraUnified::unproject(const int pyra_level, const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d)
        {
            // normalize the pixel point with K^(-1)
            double scale = 1. / pow(2, pyra_level);
            Eigen::Vector2d p2dn;
            p2dn(0) = (p2d(0) - mK(2) * scale) / (mK(0) * scale);
            p2dn(1) = (p2d(1) - mK(3) * scale) / (mK(1) * scale);

            if (mDistortion != nullptr)
            {
                mDistortion->undistort(p2dn, p2dn);
            }

            // Compute the unit ray vector that passes through the scene point.
            float rho2_u = p2dn.squaredNorm();
            float beta = 1.0 + (1.0 - m_xi * m_xi) * rho2_u;
            float lambda = (m_xi + sqrt(beta)) / (1.0 + rho2_u);

            if (beta < 0)
            {
                return false;
            }

            P3d(0) = lambda * p2dn(0);
            P3d(1) = lambda * p2dn(1);
            P3d(2) = lambda - m_xi;

            if (P3d(2) < 0)
            {
                return false;
            }

            P3d /= P3d(2);
            P3d *= z;

            return true;
        }

        void CameraUnified::projection_jacobian(Eigen::Vector3d &P3d,
                                                Eigen::Matrix<double, 2, 3> &jacobian)
        {
            assert("CameraUnified::projection_jacobian: not implementation error...\n");
        }

        std::vector<double> CameraUnified::getCameraParams()
        {
            std::vector<double> meta_data;
            // fx, fy, cx, cy, xi
            meta_data.push_back(mK(0));
            meta_data.push_back(mK(1));
            meta_data.push_back(mK(2));
            meta_data.push_back(mK(3));
            meta_data.push_back(m_xi);
            return meta_data;
        }
    } // namespace Core
} // namespace SLAM