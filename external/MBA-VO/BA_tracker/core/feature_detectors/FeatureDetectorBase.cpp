//
// Created by peidong on 2/15/20.
//

#include "FeatureDetectorBase.h"
#include <cmath>

namespace SLAM
{
    namespace Core
    {
        FeatureDetectorBase::FeatureDetectorBase(FeatureDetectorOptions::Ptr options)
            : mOptions(options)
        {
        }

        FeatureType FeatureDetectorBase::getType()
        {
            return mOptions->detector_type;
        }

        std::vector<cv::KeyPoint> &FeatureDetectorBase::getFeaturePoints(int level)
        {
            return m_featurePointsMap.at(level);
        }

        std::vector<int> &FeatureDetectorBase::getMatchedP3dIdx(int level)
        {
            return m_matchedP3dIdx.at(level);
        }

        cv::Mat &FeatureDetectorBase::getFeatureDescriptors(int level)
        {
            return m_descriptorsMap.at(level);
        }

        std::vector<cv::Point2f> &
        FeatureDetectorBase::getFeatureTrackedPosition(int level)
        {
            return m_trackedPosition.at(level);
        }

        std::vector<bool> &
        FeatureDetectorBase::getFeatureTrackFlag(int level)
        {
            return m_hasGoodTrack.at(level);
        }

        void FeatureDetectorBase::gridSelection(int im_H, int im_W, int cell_H, int cell_W)
        {
            int nCellsH = im_H / cell_H + 1;
            int nCellsW = im_W / cell_W + 1;

            for (std::map<int, std::vector<cv::KeyPoint>>::iterator it = m_featurePointsMap.begin(); it != m_featurePointsMap.end(); ++it)
            {
                int lv = it->first;
                int scale_factor = pow(2, lv);
                int im_H_lv = im_H / scale_factor;
                int im_W_lv = im_W / scale_factor;
                int cell_H_lv = cell_H / pow(1.414, lv);
                int cell_W_lv = cell_W / pow(1.414, lv);

                int nCellsH = im_H_lv / cell_H_lv + 1;
                int nCellsW = im_W_lv / cell_W_lv + 1;

                std::vector<cv::KeyPoint> &original_featurePts = it->second;
                std::vector<cv::KeyPoint> selected_featurePts(nCellsH * nCellsW);
                for (int i = 0; i < original_featurePts.size(); i++)
                {
                    int cell_index_H = original_featurePts.at(i).pt.y / cell_H_lv;
                    int cell_index_W = original_featurePts.at(i).pt.x / cell_W_lv;
                    int cell_index = cell_index_H * nCellsW + cell_index_W;

                    if (selected_featurePts.at(cell_index).response < original_featurePts.at(i).response)
                    {
                        selected_featurePts.at(cell_index) = original_featurePts.at(i);
                    }
                }

                // copy out
                original_featurePts.clear();
                for (int i = 0; i < selected_featurePts.size(); i++)
                {
                    if (selected_featurePts.at(i).response < 1e-6)
                    {
                        continue;
                    }
                    original_featurePts.push_back(selected_featurePts.at(i));
                }
            }
        }
    } // namespace Core
} // namespace SLAM