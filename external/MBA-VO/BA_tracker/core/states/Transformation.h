#ifndef __SRC_CORE_TRANSFORMATION_H_
#define __SRC_CORE_TRANSFORMATION_H_

#include <memory>

#include <Eigen/Dense>

#include "BA_tracker/core/common/Vector.h"

namespace SLAM
{
    namespace Core
    {
        class Transformation
        {
        public:
            using Ptr = std::shared_ptr<Transformation>;
            Transformation();
            Transformation(double *data);
            Transformation(const Eigen::Matrix4d &Tm);
            Transformation(const Transformation &T);
            Transformation(const Eigen::Quaterniond &R, const Eigen::Vector3d &t);

        public:
            void setTranslation(double x, double y, double z);
            void setRollPitchYaw(double roll, double pitch, double yaw);
            Eigen::Vector3d getRollPitchYaw();

        public:
            Eigen::Quaterniond getRotation();
            Eigen::Vector3d getTranslation();

        public:
            double *getData();
            const double *getData() const;
            double *getRotationData();
            const double *getRotationData() const;
            double *getTranslationData();
            const double *getTranslationData() const;

        public:
            Transformation inverse();
            Eigen::Vector3d operator*(const Eigen::Vector3d &P3d);
            Vector3d operator*(const Vector3d &P3d);
            Transformation operator*(const Transformation &T);

        public:
            Eigen::Matrix4d matrix();

            static Eigen::Matrix<double, 6, 1> log(Transformation &T);
            static Transformation exp(Eigen::Matrix<double, 6, 1> &tangent);
            static Eigen::Matrix4d hat(Eigen::Matrix<double, 6, 1> &tangent);

        private:
            // [x,y,z,qx,qy,qz,qw]
            double mInternalRepresentation[7];
        };
    } // namespace Core
} // namespace SLAM

#endif
