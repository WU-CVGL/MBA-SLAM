#include "Transformation.h"
#include "sophus/se3.hpp"

namespace SLAM
{
    namespace Core
    {
        Transformation::Transformation()
        {
            mInternalRepresentation[0] = 0;
            mInternalRepresentation[1] = 0;
            mInternalRepresentation[2] = 0;
            mInternalRepresentation[3] = 0;
            mInternalRepresentation[4] = 0;
            mInternalRepresentation[5] = 0;
            mInternalRepresentation[6] = 1;
        }

        Transformation::Transformation(double *data)
        {
            memcpy(mInternalRepresentation, data, sizeof(double) * 7);
        }

        Transformation::Transformation(const Eigen::Matrix4d &Tm)
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);
            _t = Tm.block<3, 1>(0, 3);
            _q = Eigen::Quaterniond(Tm.block<3, 3>(0, 0)).normalized();
        }

        Transformation::Transformation(const Transformation &T)
        {
            memcpy(mInternalRepresentation, T.mInternalRepresentation, 7 * sizeof(double));
        }

        Transformation::Transformation(const Eigen::Quaterniond &R, const Eigen::Vector3d &t)
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);
            _q = R.normalized();
            _t = t;
        }

        Eigen::Quaterniond Transformation::getRotation()
        {
            return Eigen::Map<Eigen::Quaterniond>(mInternalRepresentation + 3);
        }

        Eigen::Vector3d Transformation::getTranslation()
        {
            return Eigen::Map<Eigen::Vector3d>(mInternalRepresentation);
        }

        double *Transformation::getData()
        {
            return mInternalRepresentation;
        }

        const double *Transformation::getData() const
        {
            return mInternalRepresentation;
        }

        double *Transformation::getRotationData()
        {
            return mInternalRepresentation + 3;
        }

        const double *Transformation::getRotationData() const
        {
            return mInternalRepresentation + 3;
        }

        double *Transformation::getTranslationData()
        {
            return mInternalRepresentation;
        }

        const double *Transformation::getTranslationData() const
        {
            return mInternalRepresentation;
        }

        Transformation Transformation::inverse()
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);

            Eigen::Quaterniond Rinv = _q.conjugate();
            Eigen::Vector3d tinv = Rinv * (-_t);
            return Transformation(Rinv, tinv);
        }

        Eigen::Vector3d Transformation::operator*(const Eigen::Vector3d &P3d)
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);
            return _q * P3d + _t;
        }

        Vector3d Transformation::operator*(const Vector3d &P3d)
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);
            Eigen::Vector3d _P3d(P3d(0), P3d(1), P3d(2));
            Eigen::Vector3d rotated_P3d = _q * _P3d + _t;
            return Vector3d(rotated_P3d(0), rotated_P3d(1), rotated_P3d(2));
        }

        Transformation Transformation::operator*(const Transformation &T)
        {
            Eigen::Map<Eigen::Vector3d> _t(mInternalRepresentation);
            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);

            Eigen::Vector3d Tt(T.mInternalRepresentation);
            Eigen::Quaterniond Tq(T.mInternalRepresentation + 3);

            Eigen::Quaterniond R = _q * Tq;
            Eigen::Vector3d t = _q * Tt + _t;
            return Transformation(R, t);
        }

        Eigen::Vector3d Transformation::getRollPitchYaw()
        {
            double x = mInternalRepresentation[3];
            double y = mInternalRepresentation[4];
            double z = mInternalRepresentation[5];
            double w = mInternalRepresentation[6];

            double roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
            double pitch = asin(2.0 * (w * y - x * z));
            double yaw = atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));

            Eigen::Vector3d rpy(roll, pitch, yaw);
            return rpy;
        }

        void Transformation::setTranslation(double x, double y, double z)
        {
            mInternalRepresentation[0] = x;
            mInternalRepresentation[1] = y;
            mInternalRepresentation[2] = z;
        }

        void Transformation::setRollPitchYaw(double roll, double pitch, double yaw)
        {
            double cr = cos(0.5 * roll);
            double sr = sin(0.5 * roll);
            double cp = cos(0.5 * pitch);
            double sp = sin(0.5 * pitch);
            double cy = cos(0.5 * yaw);
            double sy = sin(0.5 * yaw);

            mInternalRepresentation[3] = sr * cp * cy - cr * sp * sy;
            mInternalRepresentation[4] = cr * sp * cy + sr * cp * sy;
            mInternalRepresentation[5] = cr * cp * sy - sr * sp * cy;
            mInternalRepresentation[6] = cr * cp * cy + sr * sp * sy;

            Eigen::Map<Eigen::Quaterniond> _q(mInternalRepresentation + 3);
            _q.normalize();
        }

        Eigen::Matrix<double, 6, 1> Transformation::log(Transformation &T)
        {
            Sophus::SE3d T_SE3d(T.getRotation(), T.getTranslation());
            // [translation, rotation]
            Eigen::Matrix<double, 6, 1> tangent = T_SE3d.log();
            return tangent;
        }

        Transformation Transformation::exp(Eigen::Matrix<double, 6, 1> &tangent)
        {
            // [translation, rotation]
            Sophus::SE3d T_SE3d = Sophus::SE3d::exp(tangent);
            Transformation T(T_SE3d.unit_quaternion(), T_SE3d.translation());
            return T;
        }

        Eigen::Matrix4d Transformation::hat(Eigen::Matrix<double, 6, 1> &tangent)
        {
            return Sophus::SE3d::hat(tangent);
        }

        Eigen::Matrix4d Transformation::matrix()
        {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3, 3>(0, 0) = this->getRotation().matrix();
            T.block<3, 1>(0, 3) = this->getTranslation();
            return T;
        }
    } // namespace Core
} // namespace SLAM
