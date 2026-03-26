#include "CameraBase.h"

namespace SLAM
{
    namespace Core
    {
        CameraBase::CameraBase(const Eigen::Vector4d& K, int H, int W)
            : SensorBase(), mK(K), mH(H), mW(W), mDistortion(nullptr)
        {
            computeProjectionMatrix(mK);
        }

        CameraBase::CameraBase(const Transformation& T_body_to_sensor,
                               const Eigen::Vector4d& K, int H, int W)
            : SensorBase(T_body_to_sensor), mK(K), mH(H), mW(W), mDistortion(nullptr)
        {
            computeProjectionMatrix(mK);
        }

        CameraBase::~CameraBase()
        {
            delete mDistortion;
        }

        void CameraBase::setDistortionModel(DistortionRadTan *distortion)
        {
            mDistortion = distortion;
        }

        DistortionRadTan *CameraBase::getDistortionModel()
        {
            return mDistortion;
        }

        void CameraBase::computeProjectionMatrix(const Eigen::Vector4d& K)
        {
            mKm.setIdentity();
            mKm(0, 0) = K(0);
            mKm(0, 2) = K(2);
            mKm(1, 1) = K(1);
            mKm(1, 2) = K(3);

            mKm_inv = mKm.inverse();
        }
    } // namespace Core
} // namespace SLAM