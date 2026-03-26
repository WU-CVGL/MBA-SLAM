//
// Created by peidong on 4/10/20.
//

#ifndef SLAM_CORE_POINT3D_STATIC_H
#define SLAM_CORE_POINT3D_STATIC_H

#include "Point3dBase.h"
#include <Eigen/Dense>
#include <vector>

#define MAX_CAMERAS 16

namespace SLAM
{
    namespace Core
    {
        struct Sparse3dStaticPointObservation
        {
            int frameIdx;
            // feature index within feature vectors
            int numObservations;
            int cameraIdx[MAX_CAMERAS];
            int featurePtIdx[MAX_CAMERAS];

            Sparse3dStaticPointObservation()
            {
                frameIdx = -1;
                numObservations = 0;
                std::fill(cameraIdx, cameraIdx + MAX_CAMERAS, -1);
                std::fill(featurePtIdx, featurePtIdx + MAX_CAMERAS, -1);
            };
        };

        class Point3dStatic : public Point3dBase
        {
        public:
            typedef Sparse3dStaticPointObservation Observation;
            Point3dStatic(int ptIdx, int frameIdx, Eigen::Vector3d P3d_XYZ);

        public:
            // for optimization
            void invalidate();
            double *getMutableP3dData();
            int getNumberOfObservations();

        public:
            void removeObservation(int frameIdx);

        public:
            void addPointObservation(Observation &obs);
            std::vector<Observation> &getPointObservations();

        private:
            Eigen::Vector3d mMutableP3d;
            std::vector<Observation> mObservations;
        };
    } // namespace Core
} // namespace SLAM

#endif