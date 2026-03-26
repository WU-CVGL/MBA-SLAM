#include <torch/torch.h>

#include "json/json.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/pybind11_eigen.h"
#include "pybind11/pybind11_filesystem.h"
#include "pybind11_json/pybind11_json.hpp"

#include <BA_tracker/core/sensors/CameraPinhole.h>
#include <BA_tracker/ba_tracker/blur_aware_direct_tracker.h>

#include "ba_tracker.h"
#include "factory.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::class_<pymbavo::BATracker>(m, "BATracker")
        .def(py::init<const nlohmann::json&, const Eigen::Ref<Eigen::VectorXi>>())
        .def("show", &pymbavo::BATracker::Hello, "Say Hello")
        .def("test_tensor", &pymbavo::BATracker::TestTensor, "Test tensor bindings")
        .def("init_trajectory", &pymbavo::BATracker::InitTrajectory,
             "Initialize trajectory with given start time and poses of control knots")
        .def("track", &pymbavo::BATracker::Track, "Tracking with re-localization, check if current frame is a keyframe")
        .def("track_coarse", &pymbavo::BATracker::TrackCoarse, "Coarse tracking, check if current frame is a keyframe")
        .def("set_keyframe", &pymbavo::BATracker::SetKeyframe, "Set keyframe")
        .def("temp_process_keyframe", &pymbavo::BATracker::TempProcessKeyframe, "TempProcessKeyframe: detect keypoints")
        .def("eval_poses", &pymbavo::BATracker::EvalPoses, "Eval current frame poses after tracking")
        .def("set_keyframe_keypoints", &pymbavo::BATracker::SetKeyframeKeypoints, "Set keyframe's keypoints")
        .def("get_keyframe_keypoints", &pymbavo::BATracker::GetKeyframeKeypoints, "Get keyframe's keypoints")
        .def("set_blur_frame", &pymbavo::BATracker::SetBlurFrame, "Set blur frame")
        .def("set_sharp_frame", &pymbavo::BATracker::SetSharpFrame, "Set sharp frame")
        .def("set_depth_image", &pymbavo::BATracker::SetDepthImage, "Set depth image")
        .def("set_blur_frame_from_tensor", &pymbavo::BATracker::SetBlurFrameFromTensor, "Set blur frame")
        .def("set_sharp_frame_from_tensor", &pymbavo::BATracker::SetSharpFrameFromTensor, "Set sharp frame")
        .def("set_depth_image_from_tensor", &pymbavo::BATracker::SetDepthImageFromTensor, "Set depth image")
        .def("set_depth_uncertainty", &pymbavo::BATracker::SetDepthUncertainty, "Set depth uncertainty")
        .def("set_motion_prediction", &pymbavo::BATracker::SetMotionPrediction, "Set motion prediction")
        .def("clear_motion_prediction", &pymbavo::BATracker::ClearMotionPrediction, "Clear motion prediction")
        .def("get_translation", &pymbavo::BATracker::GetTranslation, "get track result")
        .def("get_rotation", &pymbavo::BATracker::GetRotation, "get track result")
        .def("get_start_translation", &pymbavo::BATracker::GetStartTranslation, "get track result t_start")
        .def("get_start_rotation", &pymbavo::BATracker::GetStartRotation, "get track result q_start")
        .def("get_end_translation", &pymbavo::BATracker::GetEndTranslation, "get track result t_end")
        .def("get_end_rotation", &pymbavo::BATracker::GetEndRotation, "get track result q_end")
        .def("GetUnrealDepthImageCpu", &pymbavo::BATracker::GetUnrealDepthImageCpu,
            "Read depth image from Unreal dataset with given path")
        .def("GetUnrealDepthImageGpu", &pymbavo::BATracker::GetUnrealDepthImageGpu,
            "Read depth image from Unreal dataset with given path")
        .def("set_configs", &pymbavo::BATracker::SetConfigs, "set tracker configs by json")
        .def("get_configs", &pymbavo::BATracker::GetConfigs, "get tracker configs by json")
        .def("get_detector_configs", &pymbavo::BATracker::GetDetectorConfigs,
             py::return_value_policy::reference, "get detector configs")
        .def("get_detector_configs_copy", &pymbavo::BATracker::GetDetectorConfigsCopy,
                 py::return_value_policy::take_ownership, "get a copy of detector configs")
        .def("get_step_quality", &pymbavo::BATracker::GetStepQuality, "get step quality")
        ;

    py::class_<pymbavo::Factory>(m, "Factory")
        .def(py::init<>())
        .def_static("get_imageu8_from_tensor", &pymbavo::Factory::GetImageU8FromTensor,
             py::return_value_policy::move, "Get ImageU8 from tensor")
        .def_static("get_keypoints_from_frame", &pymbavo::Factory::GetKeypointsfromFrame,
             py::return_value_policy::move, "Get keypoints from frame")
        ;

    py::class_<SLAM::Core::ImagePyramid<unsigned char>, SLAM::Core::ImagePyramid<unsigned char>::Ptr>(
            m, "ImagePyramidU8")
        .def(py::init())
        .def("set_num_levels", &SLAM::Core::ImagePyramid<unsigned char>::setNumOfPyramidLevels,
             "set number of pyramid levels")
        .def("get_num_levels", &SLAM::Core::ImagePyramid<unsigned char>::getNumOfPyramidLevels,
             "get number of pyramid levels")
        .def("set_pyramid", &SLAM::Core::ImagePyramid<unsigned char>::setPyramid,
             "set pyramid with given list of images, with resolution decreasing with index")
        ;

    py::class_<SLAM::Core::Vector2d>(m, "Vector2d")
        .def(py::init())
        .def(py::init<double, double>())
        .def("__call__", py::overload_cast<int>(&SLAM::Core::Vector2d::operator()))
        ;

    py::class_<SLAM::Core::CameraPinhole, SLAM::Core::CameraPinhole::Ptr>(m, "PinholeCamera")
        .def(py::init<const Eigen::Vector4d&, int, int>())
        ;

    py::class_<SLAM::Core::Frame, SLAM::Core::Frame::Ptr>(m, "Frame")
        .def(py::init<int>())
        .def("insert_image", &SLAM::Core::Frame::insertImage, "Insert image to frame")
        .def("set_capture_time", &SLAM::Core::Frame::setCaptureTime, "Set capture time")
        .def("set_exposure_time", &SLAM::Core::Frame::setExposureTime, "Set exposure time")
        .def("set_image_pyramid", &SLAM::Core::Frame::setImagePyramid, "Set image pyramid")
        .def("compute_image_pyramid", &SLAM::Core::Frame::computeImagePyramid, "Compute image pyramid")
        .def("compute_grad_image_pyramid", &SLAM::Core::Frame::computeGradImagePyramid, "Compute grad image pyramid")
        .def("detect_features",
             py::overload_cast<int, SLAM::Core::FeatureDetectorOptions::Ptr>(&SLAM::Core::Frame::detectFeatures),
            "Detect features")
        .def("detect_features_with_mask",
             py::overload_cast<int, pymbavo::ImageFP32::Ptr, SLAM::Core::FeatureDetectorOptions::Ptr>(
                &SLAM::Core::Frame::detectFeatures),
            "Detect features with mask")
        ;

    py::class_<SLAM::Core::Image<float>, SLAM::Core::Image<float>::Ptr>(m, "ImageFP32")
        .def(py::init<int, int, int>())
        ;

    py::class_<SLAM::Core::Image<uint8_t>, SLAM::Core::Image<uint8_t>::Ptr>(m, "ImageU8")
        .def(py::init<int, int, int>())
        ;

    py::class_<SLAM::Core::FeatureDetectorOptions, SLAM::Core::FeatureDetectorOptions::Ptr>(m, "FeatureDetectorOptions")
        .def(py::init())
        .def_readwrite("score_threshold", &SLAM::Core::FeatureDetectorOptions::score_threshold)
        .def_readwrite("grid_selection_cell_H", &SLAM::Core::FeatureDetectorOptions::grid_selection_cell_H)
        .def_readwrite("grid_selection_cell_W", &SLAM::Core::FeatureDetectorOptions::grid_selection_cell_W)
        .def_readwrite("max_num_features_to_detect", &SLAM::Core::FeatureDetectorOptions::max_num_features_to_detect)
        ;

}
