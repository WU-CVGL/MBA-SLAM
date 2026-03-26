//
// Created by peidong on 2/15/20.
//

#ifndef SLAM_FEATUREDETECTORSEMIDENSE_H
#define SLAM_FEATUREDETECTORSEMIDENSE_H

#include "FeatureDetectorBase.h"
#include "BA_tracker/core/measurements/ImagePyramid.h"

namespace SLAM
{
    namespace Core
    {
        class FeatureDetectorSemiDense : public FeatureDetectorBase
        {
        public:
            FeatureDetectorSemiDense(FeatureDetectorOptions::Ptr options);

        public:
            void detect(ImagePyramid<float>::Ptr gradImageMagPyramid);
            void detect(ImagePyramid<float>::Ptr gradImageMagPyramid, Core::Image<float>::Ptr mask);
        };
    } // namespace Core
} // namespace SLAM

#endif //SLAM_FEATUREDETECTORSEMIDENSE_H
