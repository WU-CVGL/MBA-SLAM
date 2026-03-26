//
// Created by lzzhao on 12/26/23.
//

#include "factory.h"

namespace pymbavo {

ImageU8::Ptr Factory::GetImageU8FromTensor(torch::Tensor image)
{
    assert(image.dtype() == torch::kUInt8);
    CHECK_CONTIGUOUS(image);
    auto imagePtr = std::make_shared<SLAM::Core::Image<unsigned char>>();
    int H = image.sizes()[0];
    int W = image.sizes()[1];
    int C = image.sizes()[2];
    imagePtr->copyFrom(static_cast<uint8_t*>(image.data_ptr()), H, W, C);
    return imagePtr;
}

Eigen::MatrixX2f Factory::GetKeypointsfromFrame(Frame::Ptr frame, int camera_id, int level)
{
    const std::vector<cv::KeyPoint>& keypoints = frame->getFeaturePoints(camera_id, level);
    Eigen::MatrixX2f keypoints_mat(keypoints.size(), 2);
    for (int i = 0; i < keypoints.size(); ++i) {
        keypoints_mat(i, 0) = keypoints[i].pt.x;
        keypoints_mat(i, 1) = keypoints[i].pt.y;
    }
    return keypoints_mat;
}

}  // namespace pymbavo
