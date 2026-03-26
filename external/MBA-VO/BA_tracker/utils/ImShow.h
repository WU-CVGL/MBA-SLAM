//
// Created by peidong on 2/16/20.
//

#ifndef SLAM_IMSHOW_H
#define SLAM_IMSHOW_H

#include "BA_tracker/core/measurements/Image.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace SLAM
{
    namespace Utils
    {
        template <class T>
        cv::Mat convert_to_cvImage(typename Core::Image<T>::Ptr image)
        {
            int H = image->nHeight();
            int W = image->nWidth();
            int C = image->nChannels();
            int size_T = sizeof(T);

            int sourceType, targetType;
            if (C == 1 && size_T == 1)
            {
                sourceType = CV_8UC1;
                targetType = CV_8UC1;
            }
            else if (C == 1 && size_T == 4)
            {
                sourceType = CV_32FC1;
                targetType = CV_8UC1;
            }
            else if (C == 3 && size_T == 1)
            {
                sourceType = CV_8UC3;
                targetType = CV_8UC3;
            }
            else if (C == 3 && size_T == 4)
            {
                sourceType = CV_32FC3;
                targetType = CV_8UC3;
            }

            cv::Mat cvImage(cv::Size(W, H), sourceType, image->getData());
            cv::Mat cvColorImg;

            if (size_T == 4)
            {
                cv::normalize(cvImage, cvColorImg, 255, 0, cv::NORM_MINMAX, targetType);
            }
            else
            {
                cvColorImg = cvImage;
            }

            if (C == 1)
            {
                cv::cvtColor(cvColorImg, cvColorImg, cv::COLOR_GRAY2RGB);
            }

            return cvColorImg;
        }

        template <typename T>
        void get_line_end_points(Eigen::Matrix<T, 3, 1> line, int H, int W, cv::Point &pt1, cv::Point &pt2)
        {
            // find two end-points
            double x0, y0, x1, y1;
            // case: x=0
            double y = -line(2) / line(1);
            if (y > 0 && y < H)
            {
                x0 = 0;
                y0 = y;
            }
            // case: y=0
            double x = -line(2) / line(0);
            if (x > 0 && x < W)
            {
                x0 = x;
                y0 = 0;
            }
            // case: x = W-1
            y = -(line(2) + line(0) * (W - 1)) / line(1);
            if (y > 0 && y < H)
            {
                x1 = W - 1;
                y1 = y;
            }
            // case: y = H-1
            x = -(line(2) + line(1) * (H - 1)) / line(0);
            if (x > 0 && x < W)
            {
                x1 = x;
                y1 = H - 1;
            }

            pt1 = cv::Point(x0, y0);
            pt2 = cv::Point(x1, y1);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, std::vector<cv::KeyPoint> &featurePoints, float radius,
                    std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::RNG rng(12345);
            for (int i = 0; i < featurePoints.size(); ++i)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                              rng.uniform(0, 255),
                                              rng.uniform(0, 255));
                cv::circle(cvImage,
                           cv::Point(featurePoints.at(i).pt.x, featurePoints.at(i).pt.y),
                           radius,
                           color,
                           2);
            }

            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, double kpt_x, double kpt_y, float radius, std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::RNG rng(12345);

            cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                          rng.uniform(0, 255),
                                          rng.uniform(0, 255));
            cv::circle(cvImage,
                       cv::Point(kpt_x, kpt_y),
                       radius,
                       color,
                       2);

            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, std::vector<Eigen::Vector2d> &kpts, float radius, std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::RNG rng(12345);

            for (auto &kpt : kpts)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                              rng.uniform(0, 255),
                                              rng.uniform(0, 255));
                cv::circle(cvImage,
                           cv::Point(kpt(0), kpt(1)),
                           radius,
                           color,
                           2);
            }

            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, std::vector<std::vector<Eigen::Vector2d>> &all_lines, float thickness, std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::RNG rng(12345);

            for (auto &line : all_lines)
            {
                cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
                                              rng.uniform(0, 255),
                                              rng.uniform(0, 255));

                for (int i = 0; i < line.size() - 1; i++)
                {
                    cv::line(cvImage,
                             cv::Point(line.at(i)(0), line.at(i)(1)),
                             cv::Point(line.at(i + 1)(0), line.at(i + 1)(1)),
                             color,
                             thickness);
                }
            }

            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, int grid_cell_H, int grid_cell_W, std::string winName, int waitKey)
        {
            cv::Mat cvImage = convert_to_cvImage<T>(image);

            int n_cell_H = image->nHeight() / grid_cell_H + 1;
            int n_cell_W = image->nWidth() / grid_cell_W + 1;

            for (int r = 0; r < n_cell_H; r++)
            {
                cv::Scalar color = cv::Scalar(255, 0, 0);
                cv::line(cvImage, cv::Point(0, r * grid_cell_H), cv::Point(image->nWidth(), r * grid_cell_H), color);
            }

            for (int w = 0; w < n_cell_W; w++)
            {
                cv::Scalar color = cv::Scalar(255, 0, 0);
                cv::line(cvImage, cv::Point(w * grid_cell_W, 0), cv::Point(w * grid_cell_W, image->nHeight()), color);
            }
            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }

        template <class T>
        void imshow(typename Core::Image<T>::Ptr im0, const std::vector<cv::KeyPoint> &im0_keypoints,
                    typename Core::Image<T>::Ptr im1, const std::vector<cv::KeyPoint> &im1_keypoints,
                    std::vector<cv::DMatch> &matches0to1,
                    std::string winName, int waitKey)
        {
            cv::Mat cvImage0 = convert_to_cvImage<T>(im0);
            cv::Mat cvImage1 = convert_to_cvImage<T>(im1);

            cv::Mat im2;
            cv::drawMatches(cvImage0, im0_keypoints, cvImage1, im1_keypoints, matches0to1, im2);
            cv::imshow(winName, im2);
            cv::waitKey(waitKey);
        }

        // draw line: line is parameterized as "line(0) * x + line(1) * y + line(2) = 0"
        template <class T>
        void imshow(typename Core::Image<T>::Ptr image, Eigen::Vector3d line, std::string winName, int waitKey)
        {
            int H = image->nHeight();
            int W = image->nWidth();

            // find two end-points
            double x0, y0, x1, y1;
            // case: x=0
            double y = -line(2) / line(1);
            if (y > 0 && y < H)
            {
                x0 = 0;
                y0 = y;
            }
            // case: y=0
            double x = -line(2) / line(0);
            if (x > 0 && x < W)
            {
                x0 = x;
                y0 = 0;
            }
            // case: x = W-1
            y = -(line(2) + line(0) * (W - 1)) / line(1);
            if (y > 0 && y < H)
            {
                x1 = W - 1;
                y1 = y;
            }
            // case: y = H-1
            x = -(line(2) + line(1) * (H - 1)) / line(0);
            if (x > 0 && x < W)
            {
                x1 = x;
                y1 = H - 1;
            }

            // draw line
            cv::Mat cvImage = convert_to_cvImage<T>(image);
            cv::line(cvImage, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 0, 0));
            cv::imshow(winName, cvImage);
            cv::waitKey(waitKey);
        }
    } // namespace Utils
} // namespace SLAM

#endif //SLAM_VISUALIZATION_H
