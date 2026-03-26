//
// Created by peidong on 2/13/20.
//

#ifndef SLAM_FRAME_H
#define SLAM_FRAME_H

#include <map>
#include <memory>

#include "BA_tracker/core/common/Enums.h"
#include "BA_tracker/core/feature_detectors/FeatureDetectorBase.h"
#include "BA_tracker/core/sensors/CameraBase.h"
#include "Image.h"
#include "ImagePyramid.h"


namespace SLAM
{
    namespace Core
    {
        class Frame
        {
        public:
            using Ptr = std::shared_ptr<Frame>;
            Frame();
            Frame(int frameId);
            Frame(int frameId, double elapsedTimeSinceProgramStart);

        public:
            void insertImage(int cameraId, Image<unsigned char>::Ptr imagePtr);
            void copyImageFrom(int cameraId, unsigned char *dataptr, size_t H, size_t W, size_t C);

        public:
            void setCaptureTime(double captureTime);
            void setExposureTime(double exposureTime);

        public:
            Image<unsigned char>::Ptr getImagePtr(int cameraId);
            Image<unsigned char>::Ptr getImagePtr(int cameraId, int level);
            Image<float>::Ptr getGradImagePtr(int cameraId, int level);
            Image<float>::Ptr getGradImageMagPtr(int cameraId, int level);

            std::vector<cv::KeyPoint> &getFeaturePoints(int cameraId, int level);
            cv::KeyPoint &getFeaturePoint(int cameraId,
                                          int level,
                                          int featureId);

//            std::vector<size_t> getNeighborFeturePointIndices(
//                int cameraId,
//                int level,
//                double x,
//                double y,
//                double radius);

            size_t getNumFeaturePoints(int level);

            cv::Mat &getFeatureDescriptors(int cameraId, int level);

            std::vector<int> &getFeatureMatchedP3dIdx(int cameraId, int level);

            FeatureDetectorBase::Ptr getFeatureDetector(int cameraId);

            int getFrameId() const;
            double getTimestamp() const;

            double getCaptureTime();
            double getExposureTime();

        public:
            void setImagePyramid(int cameraId, ImagePyramid<unsigned char>::Ptr imagePyramid);
            void computeImagePyramid(int cameraId, int nLevels);
            void computeGradImagePyramid(int cameraId, int nLevels);
            void detectFeatures(int cameraId, Core::FeatureDetectorOptions::Ptr options);
            void detectFeatures(
                    int cameraId,
                    Core::Image<float>::Ptr mask,
                    Core::FeatureDetectorOptions::Ptr options
            );

        protected:
            // (cameraId, Image) have ownership
            std::map<int, Image<unsigned char>::Ptr> m_imagesPtrMap;

            // (cameraId, ImagePyramid) have ownership
            std::map<int, ImagePyramid<unsigned char>::Ptr> m_imagesPyramidPtrMap;

            // (cameraId, GradImagePyramid) have ownership
            std::map<int, ImagePyramid<float>::Ptr> m_gradImagePyramidPtrMap;

            // (cameraId, GradImageMagPyramid) have ownership
            std::map<int, ImagePyramid<float>::Ptr> m_gradImageMagPyramidPtrMap;

            // (cameraId, Features) have ownership
            std::map<int, FeatureDetectorBase::Ptr> m_featuresDetectorPtrMap;

            int m_frame_id;

            // in seconds
            double m_elapsedTimeSinceProgramStart;

            //
            double m_exposure_time;
            double m_capture_time;
        };
    } // namespace Core
} // namespace SLAM

#endif //SLAM_FRAME_H
