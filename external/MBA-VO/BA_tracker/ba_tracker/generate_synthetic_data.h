#ifndef SLAM_VO_GENERATE_SYNTHETIC_DATA_H
#define SLAM_VO_GENERATE_SYNTHETIC_DATA_H

#include "BA_tracker/core/common/Spline.h"
#include "BA_tracker/core/common/Vector.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace SLAM
{
    namespace VO
    {
        void synthesize_img_with_rand_shapes(const int H,
                                             const int W,
                                             cv::Mat &im);

        void warp_image(const cv::Mat &im_ref,
                        const Eigen::Quaterniond &R_c2r,
                        const Eigen::Vector3d &t_c2r,
                        const double &plane_depth,
                        const Core::VectorX<double, 4> &intrinsics,
                        cv::Mat &im_cur);

        void synthesize_motion_blurred_img(const cv::Mat &I_ref,
                                           const double &plane_depth,
                                           const Core::VectorX<double, 4> &intrinsics,
                                           Core::SplineSE3 *spline,
                                           const double capture_time,
                                           const double exposure_time,
                                           const int num_samples,
                                           cv::Mat &im_blur);
    } // namespace VO
} // namespace SLAM

#endif