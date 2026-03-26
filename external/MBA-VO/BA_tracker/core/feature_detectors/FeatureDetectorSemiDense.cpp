//
// Created by peidong on 2/15/20.
//

#include "FeatureDetectorSemiDense.h"

namespace SLAM
{
    namespace Core
    {
        FeatureDetectorSemiDense::FeatureDetectorSemiDense(FeatureDetectorOptions::Ptr options)
            : FeatureDetectorBase(options)
        {
        }

        void FeatureDetectorSemiDense::detect(
                ImagePyramid<float>::Ptr gradImageMagPyramid) {
            this->detect(gradImageMagPyramid, nullptr);
        }

        void FeatureDetectorSemiDense::detect(
            ImagePyramid<float>::Ptr gradImageMagPyramid,
            Core::Image<float>::Ptr mask
        ) {
            int nLevels = gradImageMagPyramid->getNumOfPyramidLevels();

            for (int lv = 0; lv < nLevels; ++lv)
            {
                Image<float>::Ptr gradImageMagLv = gradImageMagPyramid->getImagePtr(lv);
                int H = gradImageMagLv->nHeight();
                int W = gradImageMagLv->nWidth();
                float *gradImageMagLvPtr = gradImageMagLv->getData();

                std::vector<cv::KeyPoint> detected_features;

                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w, ++gradImageMagLvPtr)
                    {
                        if (nullptr == mask || mask->getData(h, w)[0] < 0.01)
                        {
                            if (gradImageMagLvPtr[0] > mOptions->score_threshold)
                            {
                                cv::KeyPoint feature;
                                feature.pt.x = w;
                                feature.pt.y = h;
                                feature.response = gradImageMagLvPtr[0];
                                detected_features.push_back(feature);
                            }
                        }
                    }
                }

                m_featurePointsMap.insert(
                    std::pair<int, std::vector<cv::KeyPoint>>(
                        lv,
                        detected_features));
            }

            if (mOptions->grid_selection_cell_H > 0 && mOptions->grid_selection_cell_W > 0)
            {
                this->gridSelection(
                    gradImageMagPyramid->getImagePtr(0)->nHeight(),
                    gradImageMagPyramid->getImagePtr(0)->nWidth(),
                    mOptions->grid_selection_cell_H,
                    mOptions->grid_selection_cell_W);
            }
        }
    } // namespace Core
} // namespace SLAM