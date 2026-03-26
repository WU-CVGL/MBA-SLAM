#ifndef __SRC_CORE_CAMERA_BASE_
#define __SRC_CORE_CAMERA_BASE_

#include <memory>
#include <vector>

#include "DistortionRadTan.h"
#include "SensorBase.h"
#include "BA_tracker/core/common/Enums.h"


namespace SLAM
{
    namespace Core
    {
        class CameraBase : public SensorBase
        {
        public:
            using Ptr = std::shared_ptr<CameraBase>;
            CameraBase(const Eigen::Vector4d &K, int H, int W);
            CameraBase(const Transformation &T_body_to_sensor, const Eigen::Vector4d &K, int H, int W);

            ~CameraBase();

        public:
            void setDistortionModel(DistortionRadTan *distortion);

        public:
            virtual bool project(const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d) = 0;
            virtual bool project(const int pyra_level, const Eigen::Vector3d &P3d, Eigen::Vector2d &p2d) = 0;
            virtual bool unproject(const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d) = 0;
            virtual bool unproject(const int pyra_level, const Eigen::Vector2d &p2d, const double z, Eigen::Vector3d &P3d) = 0;
            virtual std::vector<double> getCameraParams() = 0;
            virtual void projection_jacobian(Eigen::Vector3d &P3d,
                                             Eigen::Matrix<double, 2, 3> &jacobian) = 0;

        public:
            int getH() { return mH; }
            int getW() { return mW; }
            CameraType getType() { return mType; };

            const Eigen::Matrix3d &getK() { return mKm; };
            const Eigen::Matrix3d &getKinv() { return mKm_inv; };

            DistortionRadTan *getDistortionModel();

        protected:
            void computeProjectionMatrix(const Eigen::Vector4d &K);

        protected:
            Eigen::Vector4d mK; // fx, fy, cx, cy
            size_t mH;
            size_t mW;
            CameraType mType;

            // store matrix form of mK and mKinv for efficiency
            Eigen::Matrix3d mKm;
            Eigen::Matrix3d mKm_inv;

            // distortion model, has ownership
            DistortionRadTan *mDistortion;
        };
    } // namespace Core
} // namespace SLAM

#endif
