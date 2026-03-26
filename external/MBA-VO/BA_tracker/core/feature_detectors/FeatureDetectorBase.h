//
// Created by peidong on 2/15/20.
//

#ifndef SLAM_FEATUREDETECTORBASE_H
#define SLAM_FEATUREDETECTORBASE_H

#include <map>
#include <memory>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "BA_tracker/core/common/Enums.h"


namespace SLAM
{
    namespace Core
    {
        struct FeatureDetectorOptions
        {
            using Ptr = std::shared_ptr<FeatureDetectorOptions>;
            bool construct_kd_tree = false;
            bool initialize_keypoint_tracked_position = false;
            FeatureType detector_type;
            cv::ORB::ScoreType score_type;
            int max_num_features_to_detect = 1000;
            int grid_selection_cell_H = -1;
            int grid_selection_cell_W = -1;
            float score_threshold;
        };

        class FeatureDetectorBase
        {
        public:
            using Ptr = std::shared_ptr<FeatureDetectorBase>;
            FeatureDetectorBase(FeatureDetectorOptions::Ptr options);

        public:
            FeatureType getType();
            std::vector<cv::KeyPoint> &getFeaturePoints(int level);
            std::vector<int> &getMatchedP3dIdx(int level);
            cv::Mat &getFeatureDescriptors(int level);
            std::vector<cv::Point2f> &getFeatureTrackedPosition(int level);
            std::vector<bool> &getFeatureTrackFlag(int level);

        protected:
            // select feture point with maximum response in a grid cell
            void gridSelection(int im_H, int im_W, int cell_H, int cell_W);

        protected:
            FeatureDetectorOptions::Ptr mOptions;

            // (level, FeaturePoints): has ownership
            std::map<int, std::vector<cv::KeyPoint>> m_featurePointsMap;

            // (level, descriptor)
            std::map<int, cv::Mat> m_descriptorsMap;

            // index of the matched 3D point
            // (level, matched_idx)
            // -1: indicates there is no matched 3D point
            std::map<int, std::vector<int>> m_matchedP3dIdx;

            // tracked position of each keypoint
            // (level, position)
            std::map<int, std::vector<cv::Point2f>> m_trackedPosition;

            // flag to indicate if has good track
            std::map<int, std::vector<bool>> m_hasGoodTrack;
        };
    } // namespace Core
} // namespace SLAM

#endif //SLAM_FEATUREDETECTORBASE_H
