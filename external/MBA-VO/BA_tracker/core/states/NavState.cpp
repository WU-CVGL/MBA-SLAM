#include "NavState.h"

namespace SLAM
{
    namespace Core
    {
        NavState::NavState()
        {
            mVelocity.setZero();
            mBacceleration.setZero();
            mBgyro.setZero();

            mMutableVelocity.setZero();
            mMutableBacceleration.setZero();
            mMutableBgyro.setZero();
        }

        NavState::NavState(const NavState &S)
        {
            mPose = S.mPose;
            mVelocity = S.mVelocity;
            mBacceleration = S.mBacceleration;
            mBgyro = S.mBgyro;

            mMutablePose = S.mMutablePose;
            mMutableVelocity = S.mMutableVelocity;
            mMutableBacceleration = S.mMutableBacceleration;
            mMutableBgyro = S.mMutableBgyro;
        }

        Transformation NavState::getPose()
        {
            const std::lock_guard<std::mutex> lock(mStateLock);
            return mPose;
        }

        Eigen::Vector3d NavState::getVelocity()
        {
            const std::lock_guard<std::mutex> lock(mStateLock);
            return mVelocity;
        }

        Eigen::Vector3d NavState::getBiasAcc()
        {
            const std::lock_guard<std::mutex> lock(mStateLock);
            return mBacceleration;
        }

        Eigen::Vector3d NavState::getBiasGyro()
        {
            const std::lock_guard<std::mutex> lock(mStateLock);
            return mBgyro;
        }

        void NavState::setPose(Transformation T_b2w)
        {
            mPose = T_b2w;
            mMutablePose = T_b2w;
        }

        void NavState::setVelocity(Eigen::Vector3d V)
        {
            mVelocity = V;
            mMutableVelocity = V;
        }

        void NavState::setBias(Eigen::Vector3d bacc, Eigen::Vector3d bgyro)
        {
            mBacceleration = bacc;
            mBgyro = bgyro;
            mMutableBacceleration = bacc;
            mMutableBgyro = bgyro;
        }

        Transformation &NavState::getMutablePose()
        {
            return mMutablePose;
        }

        double *NavState::getMutableVelocityData()
        {
            return mMutableVelocity.data();
        }

        double *NavState::getMutableBiasAccData()
        {
            return mMutableBacceleration.data();
        }

        double *NavState::getMutableBiasGyroData()
        {
            return mMutableBgyro.data();
        }

        void NavState::invalidate()
        {
            const std::lock_guard<std::mutex> lock(mStateLock);
            mPose = mMutablePose;
            mVelocity = mMutableVelocity;
            mBacceleration = mMutableBacceleration;
            mBgyro = mMutableBgyro;
        }
    } // namespace Core
} // namespace SLAM