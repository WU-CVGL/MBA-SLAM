//
// Created by lzzhao on 12/26/23.
//

#ifndef BA_TRACKER_COMMON_H
#define BA_TRACKER_COMMON_H

#include <cassert>
#include <memory>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#include <BA_tracker/ba_tracker/blur_aware_direct_tracker.h>
#include <BA_tracker/core/measurements/Frame.h>
#include <BA_tracker/core/measurements/Image.h>
#include <BA_tracker/core/sensors/CameraPinhole.h>
#include <BA_tracker/core/states/Transformation.h>
#include <BA_tracker/utils/Geometry.h>
#include <BA_tracker/utils/ImShow.h>
#include <BA_tracker/utils/InputOutput.h>

#include "json/json.hpp"

namespace pymbavo {

namespace fs = std::filesystem;
using Camera = SLAM::Core::CameraPinhole;
using Frame = SLAM::Core::Frame;
template <typename T>
using Image = SLAM::Core::Image<T>;
using Transformation = SLAM::Core::Transformation;
using ImageU8 = SLAM::Core::Image<unsigned char>;
using ImageFP32 = SLAM::Core::Image<float>;

using DetectorCfg = SLAM::Core::FeatureDetectorOptions;
using Tracker = SLAM::VO::BlurAwareDirectTracker;
using TrackerCfg = SLAM::VO::BlurAwareDirectTrackerOptions;

using json = nlohmann::json;
using Tensor = torch::Tensor;

}

#endif //BA_TRACKER_COMMON_H
