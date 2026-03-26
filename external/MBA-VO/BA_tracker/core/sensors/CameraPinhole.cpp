//
// Created by peidong on 2/13/20.
//

#include "CameraPinhole.h"
#include <vector>

namespace SLAM
{
    namespace Core
    {
        CameraPinhole::CameraPinhole(const Eigen::Vector4d &K, int H, int W)
            : CameraBase(K, H, W)
        {
            mType = CAMERA_PINHOE;
        }

        CameraPinhole::CameraPinhole(const Transformation &T_body_to_sensor,
                                     const Eigen::Vector4d &K, int H, int W)
            : CameraBase(T_body_to_sensor, K, H, W)
        {
            mType = CAMERA_PINHOE;
        }

        bool CameraPinhole::project(const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d)
        {
            if (P3d(2) < 0)
                return false;

            Eigen::Vector2d p2dn;
            p2dn(0) = P3d(0) / (P3d(2) + 1e-8);
            p2dn(1) = P3d(1) / (P3d(2) + 1e-8);

            if (mDistortion != nullptr)
            {
                mDistortion->distort(p2dn, p2dn);
            }

            p2d(0) = mK(0) * p2dn(0) + mK(2);
            p2d(1) = mK(1) * p2dn(1) + mK(3);

            return true;
        }

        bool CameraPinhole::project(const int pyra_level, const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d)
        {
            if (P3d(2) < 0)
                return false;

            Eigen::Vector2d p2dn;
            p2dn(0) = P3d(0) / (P3d(2) + 1e-8);
            p2dn(1) = P3d(1) / (P3d(2) + 1e-8);

            if (mDistortion != nullptr)
            {
                mDistortion->distort(p2dn, p2dn);
            }

            double scale = 1. / pow(2, pyra_level);
            p2d(0) = scale * (mK(0) * p2dn(0) + mK(2));
            p2d(1) = scale * (mK(1) * p2dn(1) + mK(3));

            return true;
        }

        void CameraPinhole::projection_jacobian(Eigen::Vector3d &P3d,
                                                Eigen::Matrix<double, 2, 3> &jacobian)
        {
            double iz = 1.0f / P3d(2);
            double iz2 = iz * iz;

            jacobian(0, 0) = iz * mK(0);
            jacobian(0, 1) = 0.0f;
            jacobian(0, 2) = (-1.0f) * P3d(0) * iz2 * mK(0);
            jacobian(1, 0) = 0.0f;
            jacobian(1, 1) = iz * mK(1);
            jacobian(1, 2) = (-1.0f) * P3d(1) * iz2 * mK(1);
        }

        bool CameraPinhole::unproject(const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d)
        {
            Eigen::Vector2d p2dn;
            p2dn(0) = (p2d(0) - mK(2)) / mK(0);
            p2dn(1) = (p2d(1) - mK(3)) / mK(1);

            if (mDistortion != nullptr)
            {
                mDistortion->undistort(p2dn, p2dn);
            }

            P3d(0) = p2dn(0) * z;
            P3d(1) = p2dn(1) * z;
            P3d(2) = z;

            return true;
        }

        bool CameraPinhole::unproject(const int pyra_level, const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d)
        {
            Eigen::Vector2d p2dn;

            double scale = 1. / pow(2, pyra_level);

            p2dn(0) = (p2d(0) - mK(2) * scale) / (mK(0) * scale);
            p2dn(1) = (p2d(1) - mK(3) * scale) / (mK(1) * scale);

            if (mDistortion != nullptr)
            {
                mDistortion->undistort(p2dn, p2dn);
            }

            P3d(0) = p2dn(0) * z;
            P3d(1) = p2dn(1) * z;
            P3d(2) = z;

            return true;
        }

        std::vector<double> CameraPinhole::getCameraParams()
        {
            std::vector<double> meta_data;
            // fx, fy, cx, cy
            meta_data.push_back(mK(0));
            meta_data.push_back(mK(1));
            meta_data.push_back(mK(2));
            meta_data.push_back(mK(3));
            return meta_data;
        }
    } // namespace Core
} // namespace SLAM