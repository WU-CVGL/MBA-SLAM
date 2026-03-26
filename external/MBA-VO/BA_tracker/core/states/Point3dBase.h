//
// Created by peidong on 4/10/20.
//

#ifndef SLAM_CORE_POINT3D_BASE_H
#define SLAM_CORE_POINT3D_BASE_H

#include <Eigen/Dense>
#include <mutex>
#include <thread>

namespace SLAM
{
    namespace Core
    {
        class Point3dBase
        {
        public:
            Point3dBase(int ptIdx, int frameIdx, Eigen::Vector3d P3d_XYZ);

        public:
            int getIdx();
            int getFrameIdx();

        public:
            Eigen::Vector3d getP3d();
            virtual int getNumberOfObservations() = 0;
            virtual double *getMutableP3dData() = 0;

        public:
            virtual void removeObservation(int frameIdx) = 0;
            virtual void invalidate() = 0;

        protected:
            //
            int mPtIdx;
            // frame index which first observes current point, it should have zero sceneflow
            int mFrameIdx;
            // 3D point position
            Eigen::Vector3d mP3d;
            // mutex used for update point data from mutable data
            std::mutex mMutexUpdate;
        };
    } // namespace Core
} // namespace SLAM

#endif