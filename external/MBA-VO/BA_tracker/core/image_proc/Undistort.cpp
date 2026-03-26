#include "Undistort.h"

namespace SLAM
{
    namespace Core
    {
        Undistort::Undistort(CameraBase *from_camera, CameraBase *to_camera)
            : mFromCamera(from_camera), mToCamera(to_camera)
        {
            computePixelMappings();
        }

        Undistort::~Undistort()
        {
        }

        cv::Mat
        Undistort::undistort(const cv::Mat &image)
        {
            // undistort image
            cv::Mat uimage;
            cv::remap(image, uimage, mMapx, mMapy, cv::INTER_LINEAR);
            return uimage;
        }

        void Undistort::computePixelMappings()
        {
            size_t H = mToCamera->getH();
            size_t W = mToCamera->getW();

            mMapx = cv::Mat(H, W, CV_32FC1, -1);
            mMapy = cv::Mat(H, W, CV_32FC1, -1);

            for (int r = 0; r < mToCamera->getH(); r++)
            {
                for (int c = 0; c < mToCamera->getW(); c++)
                {
                    Eigen::Vector3d p3d;
                    if (!mToCamera->unproject(Eigen::Vector2d(c, r), 1, p3d))
                    {
                        continue;
                    }

                    Eigen::Vector2d p2d;
                    if (!mFromCamera->project(p3d, p2d))
                        continue;

                    mMapx.at<float>(r, c) = p2d(0);
                    mMapy.at<float>(r, c) = p2d(1);
                }
            }
        }

    } // namespace Core
} // namespace SLAM