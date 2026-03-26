//
// Created by peidong on 2/13/20.
//

#include "Frame.h"
#include "BA_tracker/core/feature_detectors/FeatureDetectorSemiDense.h"
#include "BA_tracker/core/image_proc/Gradient.h"

namespace SLAM
{
    namespace Core
    {
        Frame::Frame()
        {
        }

        Frame::Frame(int frameId)
        {
            m_frame_id = frameId;
        }

        Frame::Frame(int frameId, double elapsedTimeSinceProgramStart)
        {
            m_frame_id = frameId;
            m_elapsedTimeSinceProgramStart = elapsedTimeSinceProgramStart;
        }

        void Frame::insertImage(int cameraId, Image<unsigned char>::Ptr imagePtr)
        {
            m_imagesPtrMap.insert(std::pair<int, Image<unsigned char>::Ptr>(cameraId, imagePtr));
        }

        void Frame::copyImageFrom(int cameraId, unsigned char *dataptr, size_t H, size_t W, size_t C)
        {
            Image<unsigned char>::Ptr imagePtr = std::make_shared<Image<unsigned char>>();
            imagePtr->copyFrom(dataptr, H, W, C);
            insertImage(cameraId, imagePtr);
        }

        void Frame::setCaptureTime(double captureTime)
        {
            m_capture_time = captureTime;
        }

        void Frame::setExposureTime(double exposureTime)
        {
            m_exposure_time = exposureTime;
        }

        double Frame::getCaptureTime()
        {
            return m_capture_time;
        }

        double Frame::getExposureTime()
        {
            return m_exposure_time;
        }

        Image<unsigned char>::Ptr Frame::getImagePtr(int cameraId)
        {
            if (m_imagesPtrMap.count(cameraId) == 0)
                return nullptr;
            return m_imagesPtrMap.at(cameraId);
        }

        Image<unsigned char>::Ptr Frame::getImagePtr(int cameraId, int level)
        {
            return m_imagesPyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        Image<float>::Ptr Frame::getGradImagePtr(int cameraId, int level)
        {
            return m_gradImagePyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        Image<float>::Ptr Frame::getGradImageMagPtr(int cameraId, int level)
        {
            return m_gradImageMagPyramidPtrMap.at(cameraId)->getImagePtr(level);
        }

        int Frame::getFrameId() const
        {
            return m_frame_id;
        }

        double Frame::getTimestamp() const
        {
            return m_elapsedTimeSinceProgramStart;
        }

        void Frame::setImagePyramid(int cameraId, ImagePyramid<unsigned char>::Ptr imagePyramid)
        {
            insertImage(cameraId, imagePyramid->getImagePtr(0));
            m_imagesPyramidPtrMap.insert(std::pair<int, ImagePyramid<unsigned char>::Ptr>(cameraId, imagePyramid));
        }

        void Frame::computeImagePyramid(int cameraId, int nLevels)
        {
            auto imagePyramid = std::make_shared<ImagePyramid<unsigned char>>();
            imagePyramid->setNumOfPyramidLevels(nLevels);
            imagePyramid->computePyramid(m_imagesPtrMap.at(cameraId));
            setImagePyramid(cameraId, imagePyramid);
        }

        void Frame::computeGradImagePyramid(int cameraId, int nLevels)
        {
            auto gradImagePyramid = std::make_shared<ImagePyramid<float>>();
            gradImagePyramid->setNumOfPyramidLevels(nLevels);

            auto gradImageMagPyramid = std::make_shared<ImagePyramid<float>>();
            gradImageMagPyramid->setNumOfPyramidLevels(nLevels);

            for (int lv = 0; lv < nLevels; ++lv)
            {
                Image<unsigned char>::Ptr imagePtr = m_imagesPyramidPtrMap.at(cameraId)->getImagePtr(lv);

                int H = imagePtr->nHeight();
                int W = imagePtr->nWidth();
                int C = imagePtr->nChannels();

                auto gradImage = std::make_shared<Image<float>>(H, W, 2 * C);
                auto gradImageMag = std::make_shared<Image<float>> (H, W, 1);

                compute_image_gradients<unsigned char, float>(imagePtr, gradImage, gradImageMag);

                gradImagePyramid->copyPyramidFrom(lv, gradImage->getData(), H, W, 2 * C);
                gradImageMagPyramid->copyPyramidFrom(lv, gradImageMag->getData(), H, W, 1);
            }

            m_gradImagePyramidPtrMap.insert(std::pair<int, ImagePyramid<float>::Ptr>(cameraId, gradImagePyramid));
            m_gradImageMagPyramidPtrMap.insert(std::pair<int, ImagePyramid<float>::Ptr>(cameraId, gradImageMagPyramid));
        }

        void Frame::detectFeatures(int cameraId, Core::FeatureDetectorOptions::Ptr options)
        {
            this->detectFeatures(cameraId, nullptr, options);
        }

        void Frame::detectFeatures(int cameraId, Core::Image<float>::Ptr mask, Core::FeatureDetectorOptions::Ptr options)
        {
            switch (options->detector_type)
            {
            case ENUM_SEMIDENSE:
            {
                auto featureDetector =
                    std::make_shared<FeatureDetectorSemiDense>(options);

                featureDetector->detect(
                    m_gradImageMagPyramidPtrMap.at(cameraId), mask);

                m_featuresDetectorPtrMap.insert(
                    std::pair<int, FeatureDetectorSemiDense::Ptr>(
                        cameraId, featureDetector));

                break;
            }
            case ENUM_SPARSE_ORB:
            case ENUM_SPARSE_SHI_TOMASI:
            default:
            {
                printf("%s:L%d not implemented detector_type %d!\n", __FILE__, __LINE__, options->detector_type);
                assert(0);
            }
            }
        }

        std::vector<cv::KeyPoint> &Frame::getFeaturePoints(int cameraId, int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeaturePoints(level);
        }

        cv::Mat &Frame::getFeatureDescriptors(int cameraId, int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeatureDescriptors(level);
        }

        std::vector<int> &Frame::getFeatureMatchedP3dIdx(int cameraId,
                                                         int level)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getMatchedP3dIdx(level);
        }

        FeatureDetectorBase::Ptr Frame::getFeatureDetector(int cameraId)
        {
            return m_featuresDetectorPtrMap.at(cameraId);
        }

        cv::KeyPoint &Frame::getFeaturePoint(int cameraId, int level, int featureId)
        {
            return m_featuresDetectorPtrMap.at(cameraId)->getFeaturePoints(level).at(featureId);
        }

//        std::vector<size_t> Frame::getNeighborFeturePointIndices(int cameraId, int level, double x, double y, double radius)
//        {
//            return static_cast<FeatureDetectorSparse *>(m_featuresDetectorPtrMap.at(cameraId))->get_neighbor_feturePoint_indices(x, y, radius);
//        }

        size_t Frame::getNumFeaturePoints(int level)
        {
            size_t num_keypoints = 0;
            for (auto it = m_featuresDetectorPtrMap.begin(); it != m_featuresDetectorPtrMap.end(); it++)
            {
                num_keypoints += it->second->getFeaturePoints(level).size();
            }
            return num_keypoints;
        }
    } // namespace Core
} // namespace SLAM