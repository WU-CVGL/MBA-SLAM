#include "ba_tracker.h"

namespace pymbavo {

BATracker::BATracker(const json& configs, const Eigen::Ref<Eigen::VectorXi> pattern)
{
    SetConfigs(configs);
    SetPattern(pattern);

    options_.camera_id = 0;

    const Eigen::Vector4d intrinsics(options_.intrinsics.values[0],
                                     options_.intrinsics.values[1],
                                     options_.intrinsics.values[2],
                                     options_.intrinsics.values[3]);
    camera_ = std::make_shared<SLAM::Core::CameraPinhole>(
        intrinsics, options_.im_size_HW.values[0], options_.im_size_HW.values[1]);
    tracker_ = std::make_shared<SLAM::VO::BlurAwareDirectTracker>(options_);
    tracker_->setCamera(camera_);
}

void BATracker::Hello() const { std::cout << "Hello" << std::endl; }

torch::Tensor BATracker::TestTensor()
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor test = torch::zeros({2, 3, 4}, options);
    return test;
}

double BATracker::GetStepQuality() const { return tracker_->getStepQuality(); }

torch::Tensor BATracker::GetUnrealDepthImageCpu(const fs::path& image_path) const
{
    auto depth_ray = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    SLAM::Utils::load_depthMap(image_path, depth_ray);

    auto depth_image = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    SLAM::Utils::convert_ray_d_to_z(camera_, depth_ray, depth_image);
    if (options_.with_gui) {
        SLAM::Utils::imshow<float>(depth_image, "Depth Image (CXX)", 2);
    }

    // cv::Mat depth_image_cv(options_.im_size_HW.values[0], options_.im_size_HW.values[1], CV_32F, depth_image.getData());

    // fs::path output_image("/tmp/test.png");
    // std::cout << "Writing depth_image to " << output_image << std::endl;
    // cv::imwrite(output_image.string(), depth_image_cv);

    std::cout << "Copying depth_image to torch::Tensor (CPU)..." << std::endl;
    auto image_tensor = torch::from_blob(
        depth_image->getData(),
        {options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    return image_tensor.clone();
}

torch::Tensor BATracker::GetUnrealDepthImageGpu(const fs::path& image_path) const
{
    auto depth_ray = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    SLAM::Utils::load_depthMap(image_path, depth_ray);

    auto depth_image = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    SLAM::Utils::convert_ray_d_to_z(camera_, depth_ray, depth_image);

    std::cout << "Uploading depth_image to GPU" << std::endl;
    depth_image->uploadToGpu();

    std::cout << "Copying depth_image to torch::Tensor (GPU)..." << std::endl;

    auto image_tensor = torch::from_blob(
        depth_image->getGpuData(),
        {options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0));

    return image_tensor;
}

void BATracker::InitTrajectory(float cur_start_time,
                               const Eigen::Vector4d& q_ref,
                               const Eigen::Vector4d& q_cur_start,
                               const Eigen::Vector4d& q_cur_end,
                               const Eigen::Vector3d& t_ref,
                               const Eigen::Vector3d& t_cur_start,
                               const Eigen::Vector3d& t_cur_end)
{
    auto spline = tracker_->getSplineTrajectory();
    spline->setSamplingInterval(options_.dt_ctrl_knot);
    std::cout << "SetInitPose: options_.dt_ctrl_knot = " << options_.dt_ctrl_knot
              << std::endl;
    spline->setSplineDegK(options_.spline_deg_k);
    std::cout << "SetInitPose: options_.spline_deg_k = " << options_.spline_deg_k
              << std::endl;
    spline->setStartTime(cur_start_time);
    spline->InsertControlKnot(Eigen::Quaterniond(q_cur_start), t_cur_start);
    spline->InsertControlKnot(Eigen::Quaterniond(q_cur_end), t_cur_end);
    Transformation T_ref2w(Eigen::Quaterniond(q_ref), t_ref);
    Transformation T_w2ref = T_ref2w.inverse();
    spline->TransformBy(T_w2ref.getRotation(), T_w2ref.getTranslation());
}

void BATracker::SetBlurFrame(Frame::Ptr frame) {
    blur_frame_ = frame;
}

void BATracker::SetSharpFrame(Frame::Ptr frame) {
    sharp_frame_ = frame;
}

void BATracker::SetDepthImage(Image<float>::Ptr image) {
    depth_image_ = image;
}

void BATracker::SetBlurFrameFromTensor(int frame_id,
                                     Tensor image,
                                     double capture_time,
                                     double exposure_time)
{
    CHECK_CONTIGUOUS(image);
    assert(image.dtype() == torch::kUInt8);
    blur_frame_ = std::make_shared<Frame>(frame_id);
    int H = image.sizes()[0];
    int W = image.sizes()[1];
    int C = image.sizes()[2];
    blur_frame_->copyImageFrom(
        options_.camera_id, static_cast<uint8_t*>(image.data_ptr()), H, W, C);
    blur_frame_->setCaptureTime(capture_time);
    blur_frame_->setExposureTime(exposure_time);
    blur_frame_->computeImagePyramid(options_.camera_id, options_.num_pyramid_levels);
    blur_frame_->computeGradImagePyramid(options_.camera_id, options_.num_pyramid_levels);
}

void BATracker::SetSharpFrameFromTensor(int frame_id, Tensor image, double capture_time)
{
    CHECK_CONTIGUOUS(image);
    assert(image.dtype() == torch::kUInt8);
    sharp_frame_ = std::make_shared<Frame>(frame_id);
    int H = image.sizes()[0];
    int W = image.sizes()[1];
    int C = image.sizes()[2];
    sharp_frame_->copyImageFrom(
        options_.camera_id, static_cast<uint8_t*>(image.data_ptr()), H, W, C);
    sharp_frame_->setCaptureTime(capture_time);
    sharp_frame_->computeImagePyramid(options_.camera_id, options_.num_pyramid_levels);
    sharp_frame_->computeGradImagePyramid(options_.camera_id, options_.num_pyramid_levels);
}

void BATracker::SetDepthImageFromTensor(Tensor input)
{
    CHECK_CONTIGUOUS(input);
    int H = input.sizes()[0];
    int W = input.sizes()[1];
    int C = input.sizes()[2];
    assert(1 == C);
    assert(options_.im_size_HW.values[0] == H);
    assert(options_.im_size_HW.values[1] == W);
    depth_image_ = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    depth_image_->copyFrom(static_cast<float*>(input.data_ptr()), H, W, 1);
    if (options_.with_gui) {
        SLAM::Utils::imshow<float>(depth_image_, "Depth Image (CXX)", 2);
    }
}

void BATracker::SetDepthUncertainty(Tensor input)
{
    CHECK_CONTIGUOUS(input);
    int H = input.sizes()[0];
    int W = input.sizes()[1];
    int C = input.sizes()[2];
    assert(1 == C);
    assert(options_.im_size_HW.values[0] == H);
    assert(options_.im_size_HW.values[1] == W);
    depth_uncertainty_ = std::make_shared<Image<float>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    depth_uncertainty_->copyFrom(static_cast<float*>(input.data_ptr()), H, W, 1);
    if (options_.with_gui) {
        SLAM::Utils::imshow<float>(depth_uncertainty_, "Depth Uncertainty (CXX)", 2);
    }
}

void BATracker::SetMotionPrediction(const Eigen::Matrix4d& pose)
{
    motion_predition_ = std::make_shared<Transformation>(pose);
}

void BATracker::ClearMotionPrediction() { motion_predition_ = nullptr; }

void BATracker::SetKeyframeKeypoints(int level,
                                     const Eigen::MatrixX2d& keypoints_xy,
                                     const Eigen::VectorXd& keypoints_z)
{
    tracker_->mOptions.num_keypoints[level] = keypoints_z.size();
    printf("set_keyframe_keypoints: %d at level %d\n",
           tracker_->mOptions.num_keypoints[level],
           level);
    std::vector<SLAM::Core::Vector2d> tmp_keypoints_xy;
    tmp_keypoints_xy.reserve(keypoints_z.size());
    for (auto row : keypoints_xy.rowwise()) {
        tmp_keypoints_xy.emplace_back(SLAM::Core::Vector2d(row[0], row[1]));
    }
    tracker_->mOptions.tmp_keypoints_xy_pybind[level] =
        std::make_shared<decltype(tmp_keypoints_xy)>(tmp_keypoints_xy);
    tracker_->mOptions.tmp_keypoints_z_pybind[level] =
        std::make_shared<Eigen::VectorXd>(keypoints_z);
    tracker_->mOptions.tmp_keypoints_xy[level] =
        tracker_->mOptions.tmp_keypoints_xy_pybind[level]->data();
    tracker_->mOptions.tmp_keypoints_z[level] =
        tracker_->mOptions.tmp_keypoints_z_pybind[level]->data();
}

Eigen::MatrixX3d BATracker::GetKeyframeKeypoints(int level)
{
    Eigen::MatrixX3d keypoints_xyz(tracker_->mOptions.num_keypoints[level], 3);
    for (int i = 0; i < tracker_->mOptions.num_keypoints[level]; i++) {
        keypoints_xyz(i, 0) = tracker_->mOptions.tmp_keypoints_xy[level][i](0);
        keypoints_xyz(i, 1) = tracker_->mOptions.tmp_keypoints_xy[level][i](1);
        keypoints_xyz(i, 2) = tracker_->mOptions.tmp_keypoints_z[level][i];
    }
    return keypoints_xyz;
}

template <typename T>
void BATracker::TensorToImage(Tensor input, Image<T>* output)
{
    CHECK_CONTIGUOUS(input);
    int H = input.sizes()[0];
    int W = input.sizes()[1];
    int C = input.sizes()[2];
    assert(1 == C);
    assert(options_.im_size_HW.values[0] == H);
    assert(options_.im_size_HW.values[1] == W);
    output = std::make_shared<Image<T>>(
        options_.im_size_HW.values[0], options_.im_size_HW.values[1], 1);
    output->copyFrom(static_cast<T*>(input.data_ptr()), H, W, 1);
}

bool BATracker::Track()
{
    bool isKeyframe = tracker_->track(blur_frame_);
    return isKeyframe;
}

bool BATracker::TrackCoarse()
{
    auto [isKeyframe, failure_indicators] =
        tracker_->trackCoarse(blur_frame_, motion_predition_);
    return isKeyframe;
}

void BATracker::SetKeyframe() { tracker_->setKeyframe(sharp_frame_); }

void BATracker::EvalPoses(bool isKeyframe)
{
    const auto [T_start, T_b2w, T_end] = tracker_->evalPoses(isKeyframe, blur_frame_);
    T_b2w_start_ = T_start;
    T_b2w_ = T_b2w;
    T_b2w_end_ = T_end;
}

void BATracker::TempProcessKeyframe()
{
    tracker_->tmpProcessKeyframe(sharp_frame_, depth_image_, depth_uncertainty_);
}

Eigen::Vector4d BATracker::GetRotation() { return T_b2w_.getRotation().coeffs(); }

Eigen::Vector3d BATracker::GetTranslation() { return T_b2w_.getTranslation(); }

Eigen::Vector4d BATracker::GetStartRotation()
{
    return T_b2w_start_.getRotation().coeffs();
}

Eigen::Vector3d BATracker::GetStartTranslation() { return T_b2w_start_.getTranslation(); }

Eigen::Vector4d BATracker::GetEndRotation() { return T_b2w_end_.getRotation().coeffs(); }

Eigen::Vector3d BATracker::GetEndTranslation() { return T_b2w_end_.getTranslation(); }

void BATracker::SetConfigs(const json& configs)
{
    // spline trajectory
    if (configs.contains("dt_ctrl_knot")) {
        options_.dt_ctrl_knot = configs["dt_ctrl_knot"];
    }
    if (configs.contains("num_init_ctrl_knots")) {
        options_.spline_deg_k = configs["num_init_ctrl_knots"];
    }
    if (configs.contains("dt_frame")) {
        options_.dt_frame = configs["dt_frame"];
    }

    // image intrinsics
    if (configs.contains("H")) {
        options_.im_size_HW.values[0] = configs["H"];
    }
    if (configs.contains("W")) {
        options_.im_size_HW.values[1] = configs["W"];
    }
    if (configs.contains("fx")) {
        options_.intrinsics.values[0] = configs["fx"];
    }
    if (configs.contains("fy")) {
        options_.intrinsics.values[1] = configs["fy"];
    }
    if (configs.contains("cx")) {
        options_.intrinsics.values[2] = configs["cx"];
    }
    if (configs.contains("cy")) {
        options_.intrinsics.values[3] = configs["cy"];
    }

    // detector
    if (configs.contains("score_threshold")) {
        options_.score_threshold = configs["score_threshold"];
    }
    if (configs.contains("grid_selection_cell_H")) {
        options_.grid_selection_cell_H = configs["grid_selection_cell_H"];
    }
    if (configs.contains("grid_selection_cell_W")) {
        options_.grid_selection_cell_W = configs["grid_selection_cell_W"];
    }
    if (configs.contains("max_num_features_to_detect")) {
        options_.max_num_features_to_detect = configs["max_num_features_to_detect"];
    }

    // solver
    if (configs.contains("huber_k")) {
        options_.huber_k = configs["huber_k"];
    }
    if (configs.contains("solver_type")) {
        options_.solver_type = configs["solver_type"];
    }

    //
    if (configs.contains("num_pyramid_level")) {
        options_.num_pyramid_levels = configs["num_pyramid_level"];
    }
    if (configs.contains("num_virtual_images_per_frame")) {
        options_.num_virtual_poses_per_frame[0] =
            configs["num_virtual_images_per_frame"]["value0"];
        options_.num_virtual_poses_per_frame[1] =
            configs["num_virtual_images_per_frame"]["value1"];
        options_.num_virtual_poses_per_frame[2] =
            configs["num_virtual_images_per_frame"]["value2"];
        options_.num_virtual_poses_per_frame[3] =
            configs["num_virtual_images_per_frame"]["value3"];
        options_.num_virtual_poses_per_frame[4] =
            configs["num_virtual_images_per_frame"]["value4"];
        options_.num_virtual_poses_per_frame[5] =
            configs["num_virtual_images_per_frame"]["value5"];
    }

    // storage
    options_.max_num_frames = 1;
    options_.max_num_virtual_sharp_imgs_per_frame = 64;
    options_.max_num_keypoints = 500;
    options_.max_patch_size = 121;
    if (configs.contains("max_num_frames")) {
        options_.max_num_frames = configs["max_num_frames"];
    }
    if (configs.contains("max_num_virtual_sharp_imgs_per_frame")) {
        options_.max_num_virtual_sharp_imgs_per_frame =
            configs["max_num_virtual_sharp_imgs_per_frame"];
    }
    if (configs.contains("max_num_keypoints")) {
        options_.max_num_keypoints = configs["max_num_keypoints"];
    }
    if (configs.contains("max_patch_size")) {
        options_.max_patch_size = configs["max_patch_size"];
    }
    if (configs.contains("max_num_ctrl_knots")) {
        options_.max_num_ctrl_knots = configs["max_num_ctrl_knots"];
    }

    // keyframe creation
    if (configs.contains("max_chi_square_error")) {
        options_.max_chi_square_error = configs["max_chi_square_error"];
    }
    if (configs.contains("keyframe_max_flow_mag0")) {
        options_.keyframe_max_flow_mag0 = configs["keyframe_max_flow_mag0"];
    }
    if (configs.contains("keyframe_max_flow_mag1")) {
        options_.keyframe_max_flow_mag1 = configs["keyframe_max_flow_mag1"];
    }
    if (configs.contains("keyframe_max_flow_mag2")) {
        options_.keyframe_max_flow_mag2 = configs["keyframe_max_flow_mag2"];
    }
    if (configs.contains("keyframe_max_blur_kernel_mag")) {
        options_.keyframe_max_blur_kernel_mag = configs["keyframe_max_blur_kernel_mag"];
    }

    // imshow
    if (configs.contains("with_gui")) {
        options_.with_gui = configs["with_gui"];
    }
    if (configs.contains("frame_id_start_to_pause_imshow")) {
        options_.frame_id_start_to_pause_imshow =
            configs["frame_id_start_to_pause_imshow"];
    }
    if (configs.contains("line_thickness")) {
        options_.line_thickness = configs["line_thickness"];
    }
}

json BATracker::GetConfigs() const
{
    return {{"dt_ctrl_knot", options_.dt_ctrl_knot},
            {"num_init_ctrl_knots", options_.spline_deg_k},
            {"dt_frame", options_.dt_frame},
            {"H", options_.im_size_HW.values[0]},
            {"W", options_.im_size_HW.values[1]},
            {"fx", options_.intrinsics.values[0]},
            {"fy", options_.intrinsics.values[1]},
            {"cx", options_.intrinsics.values[2]},
            {"cy", options_.intrinsics.values[3]},
            {"score_threshold", options_.score_threshold},
            {"grid_selection_cell_H", options_.grid_selection_cell_H},
            {"grid_selection_cell_W", options_.grid_selection_cell_W},
            {"max_num_features_to_detect", options_.max_num_features_to_detect},
            {"huber_k", options_.huber_k},
            {"solver_type", options_.solver_type},
            {"num_pyramid_level", options_.num_pyramid_levels},
            {"num_virtual_images_per_frame",
             json({
                 {"value0", options_.num_virtual_poses_per_frame[0]},
                 {"value1", options_.num_virtual_poses_per_frame[1]},
                 {"value2", options_.num_virtual_poses_per_frame[2]},
                 {"value3", options_.num_virtual_poses_per_frame[3]},
                 {"value4", options_.num_virtual_poses_per_frame[4]},
                 {"value5", options_.num_virtual_poses_per_frame[5]},
             })},
            {"max_num_frames", options_.max_num_frames},
            {"max_num_virtual_sharp_imgs_per_frame",
             options_.max_num_virtual_sharp_imgs_per_frame},
            {"max_num_keypoints", options_.max_num_keypoints},
            {"max_patch_size", options_.max_patch_size},
            {"max_num_ctrl_knots", options_.max_num_ctrl_knots},
            {"max_chi_square_error", options_.max_chi_square_error},
            {"keyframe_max_flow_mag0", options_.keyframe_max_flow_mag0},
            {"keyframe_max_flow_mag1", options_.keyframe_max_flow_mag1},
            {"keyframe_max_flow_mag2", options_.keyframe_max_flow_mag2},
            {"keyframe_max_blur_kernel_mag", options_.keyframe_max_blur_kernel_mag},
            {"with_gui", options_.with_gui},
            {"frame_id_start_to_pause_imshow", options_.frame_id_start_to_pause_imshow},
            {"line_thickness", options_.line_thickness}};
}

void BATracker::SetPattern(const Eigen::VectorXi& pattern)
{
    int len = pattern.size();
    int half_len = len / 2;
    options_.patch_size[0] = half_len;
    options_.patch_size[1] = half_len;
    options_.patch_size[2] = half_len;
    options_.patch_size[3] = half_len;
    options_.patch_size[4] = half_len;
    options_.patch_size[5] = half_len;

    for (int lvl = 0; lvl < options_.num_pyramid_levels; lvl++) {
        options_.local_patch_pattern_xy[lvl] = new int[len];
        for (int i = 0; i < len; i++) {
            options_.local_patch_pattern_xy[lvl][i] = pattern[i];
        }
    }
}

void BATracker::LoadGroundTruth(const fs::path& path_to_file,
                                std::vector<std::array<double, 8>>& gt_poses)
{
    std::ifstream fileReader;
    fileReader.open(path_to_file, std::ifstream::in);
    if (!fileReader.is_open()) {
        printf("%s fails to load %s, quit...\n", __FUNCTION__, path_to_file.c_str());
        std::exit(0);
    }

    while (!fileReader.eof()) {
        std::string line;
        std::getline(fileReader, line);
        if (line.find('#') != std::string::npos || line.empty()) continue;

        std::stringstream line_stream(line);

        std::array<double, 8> pose;
        line_stream >> pose[0] >> pose[1] >> pose[2] >> pose[3] >> pose[4] >> pose[5] >>
            pose[6] >> pose[7];
        gt_poses.push_back(pose);
    }
}

std::array<double, 8> BATracker::GetPose(
    const std::vector<std::array<double, 8>>& gt_poses,
    double t)
{
    for (auto& pose : gt_poses) {
        if (std::fabs(pose[0] - t) < 0.0099) {
            return pose;
        }
    }
    assert(false);
}

DetectorCfg::Ptr BATracker::GetDetectorConfigs() const
{
    return tracker_->getDetectorOptions();
}

DetectorCfg BATracker::GetDetectorConfigsCopy() const
{
    DetectorCfg cfg;
    const DetectorCfg::Ptr tracker_cfg = tracker_->getDetectorOptions();

    cfg.construct_kd_tree = tracker_cfg->construct_kd_tree;
    cfg.initialize_keypoint_tracked_position = tracker_cfg->initialize_keypoint_tracked_position;
    cfg.detector_type = tracker_cfg->detector_type;
    cfg.score_type = tracker_cfg->score_type;
    cfg.max_num_features_to_detect = tracker_cfg->max_num_features_to_detect;
    cfg.grid_selection_cell_H = tracker_cfg->grid_selection_cell_H;
    cfg.grid_selection_cell_W = tracker_cfg->grid_selection_cell_W;
    cfg.score_threshold = tracker_cfg->score_threshold;
    return cfg;
}

}  // namespace pymbavo
