//
// Created by peidong on 4/10/20.
//

#ifndef SLAM_CORE_POINT3D_DYNAMIC_H
#define SLAM_CORE_POINT3D_DYNAMIC_H

#include "Point3dBase.h"
#include "BA_tracker/core/common/Enums.h"
#include <Eigen/Dense>
#include <vector>

#define MAX_CAMERAS 16

namespace SLAM
{
    namespace Core
    {
        struct Sparse3dDynamicPointObservation
        {
            int frameIdx;
            // feature index within feature vectors
            int numObservations;
            int cameraIdx[MAX_CAMERAS];
            int featurePtIdx[MAX_CAMERAS];
            MotionStatus motionStatus;

            Eigen::Vector3d scene_flow;

            Sparse3dDynamicPointObservation()
            {
                frameIdx = -1;
                numObservations = 0;
                std::fill(cameraIdx, cameraIdx + MAX_CAMERAS, -1);
                std::fill(featurePtIdx, featurePtIdx + MAX_CAMERAS, -1);
                scene_flow.setZero();
                motionStatus = MotionStatus::UNCERTAIN;
            };
        };

        class Point3dDynamic : public Point3dBase
        {
        public:
            typedef Sparse3dDynamicPointObservation Observation;
            Point3dDynamic(int ptIdx,
                           int frameIdx,
                           Eigen::Vector3d P3d_XYZ,
                           MotionStatus &status,
                           double creation_time);

        public:
            int getNumberOfObservations();
            Eigen::Vector3d getSceneFlow();

            // for optimization
            double *getMutableP3dData();
            double *getMutableSceneFlow();

            void invalidate();
            void removeObservation(int frameIdx);

        public:
            void addPointObservation(Observation &obs);
            std::vector<Observation> &getPointObservations();

        public:
            double getAge(double current_time);
            void updateMotionStatus(MotionStatus &status, double current_time);
            MotionStatus &getMotionStatus();

        private:
            // Scene flow which is used to track against latest frame
            Eigen::Vector3d mSceneFlow;
            Eigen::Vector3d mMutableSceneFlow;

            // KFSceneFlows do not contain the one for the keyframe
            // which first observes the point, since it always has zero sceneflow
            std::vector<Eigen::Vector3d> mKfSceneFlows;
            std::vector<Eigen::Vector3d> mMutableP3d_KfSceneFlows;

            // We assume the first observation always corresponding to the keyframe which first observes the point
            std::vector<Observation> mObservations;

            //
            MotionStatus mMotionStatus;
            double mMotionStatusUpdateTime;
        };
    } // namespace Core
} // namespace SLAM

#endif