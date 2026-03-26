//
// Created by peidong on 4/10/20.
//

#ifndef SLAM_NAVSTATE_H
#define SLAM_NAVSTATE_H

#include "Transformation.h"
#include "BA_tracker/core/common/RadToDeg.h"
#include <mutex>
#include <thread>

namespace SLAM
{
    namespace Core
    {
        class NavState
        {
        public:
            NavState();
            NavState(const NavState &S);

        public:
            Transformation getPose();
            Eigen::Vector3d getVelocity();
            Eigen::Vector3d getBiasAcc();
            Eigen::Vector3d getBiasGyro();

        public:
            void setPose(Transformation T_b2w);
            void setVelocity(Eigen::Vector3d V);
            void setBias(Eigen::Vector3d bacc, Eigen::Vector3d bgyro);

        public:
            Transformation &getMutablePose();
            double *getMutableVelocityData();
            double *getMutableBiasAccData();
            double *getMutableBiasGyroData();

        public:
            void invalidate();

        public:
            friend std::ostream &operator<<(std::ostream &os, NavState &state)
            {
                Eigen::Vector3d rpy = state.getPose().getRollPitchYaw();
                rpy(0) = rad_to_deg(rpy(0));
                rpy(1) = rad_to_deg(rpy(1));
                rpy(2) = rad_to_deg(rpy(2));

                os << "R: " << rpy.transpose() << "\n"
                   << "t: " << state.getPose().getTranslation().transpose() << "\n"
                   << "V: " << state.getVelocity().transpose() << "\n"
                   << "bacc: " << state.getBiasAcc().transpose() << "\n"
                   << "bgyro: " << state.getBiasGyro().transpose() << "\n";

                return os;
            }

        private:
            // Transformation matrices are defined from body frame to global world frame
            // Vehicle states after optimization
            Transformation mPose;
            Eigen::Vector3d mVelocity;
            Eigen::Vector3d mBacceleration;
            Eigen::Vector3d mBgyro;

            // Vehicle states for optimization
            Transformation mMutablePose;
            Eigen::Vector3d mMutableVelocity;
            Eigen::Vector3d mMutableBacceleration;
            Eigen::Vector3d mMutableBgyro;

            std::mutex mStateLock;
        };
    } // namespace Core
} // namespace SLAM

#endif