#pragma once

#include "common.h"

namespace pymbavo {

class BATracker {
   public:
    BATracker(const json& configs, const Eigen::Ref<Eigen::VectorXi> pattern);
    Tensor TestTensor();
    double GetStepQuality() const;
    Eigen::Vector3d GetTranslation();
    Eigen::Vector4d GetRotation();
    Eigen::Vector3d GetStartTranslation();
    Eigen::Vector4d GetStartRotation();
    Eigen::Vector3d GetEndTranslation();
    Eigen::Vector4d GetEndRotation();
    Tensor GetUnrealDepthImageCpu(const fs::path& image_path) const;
    Tensor GetUnrealDepthImageGpu(const fs::path& image_path) const;
    void Hello() const;
    bool Track();
    bool TrackCoarse();
    void TempProcessKeyframe();
    void SetKeyframe();
    void EvalPoses(bool isKeyframe);

    void SetConfigs(const json& configs);
    json GetConfigs() const;
    DetectorCfg::Ptr GetDetectorConfigs() const;
    DetectorCfg GetDetectorConfigsCopy() const;

    void SetPattern(const Eigen::VectorXi& pattern);

    void InitTrajectory(float cur_start_time,
                        const Eigen::Vector4d& q_ref,
                        const Eigen::Vector4d& q_cur_start,
                        const Eigen::Vector4d& q_cur_end,
                        const Eigen::Vector3d& t_ref,
                        const Eigen::Vector3d& t_cur_start,
                        const Eigen::Vector3d& t_cur_end);

    void SetBlurFrame(Frame::Ptr frame);

    void SetSharpFrame(Frame::Ptr frame);

    void SetDepthImage(Image<float>::Ptr image);

    void SetBlurFrameFromTensor(int frame_id,
                      Tensor image,
                      double capture_time,
                      double exposure_time);

    void SetSharpFrameFromTensor(int frame_id,
                                Tensor image,
                                double capture_time);

    void SetDepthImageFromTensor(Tensor input);

    void SetDepthUncertainty(Tensor input);

    // This function will replace BlurAwareDirectTracker::tmpProcessKeyframe
    void SetKeyframeKeypoints(int level,
                              const Eigen::MatrixX2d& keypoints_xy,
                              const Eigen::VectorXd& keypoints_z);

    Eigen::MatrixX3d GetKeyframeKeypoints(int level);

    void SetMotionPrediction(const Eigen::Matrix4d& pose);
    void ClearMotionPrediction();

   private:
    void LoadOptions();
    void LoadPattern();

   private:
    template <typename T>
    void TensorToImage(Tensor input, Image<T>* output);

    void LoadGroundTruth(const fs::path& path_to_file,
                         std::vector<std::array<double, 8>>& gt_poses);

    std::array<double, 8> GetPose(const std::vector<std::array<double, 8>>& gt_poses,
                                  double t);

   private:

    Transformation T_b2w_;
    Transformation T_b2w_start_;
    Transformation T_b2w_end_;
    Image<float>::Ptr depth_image_;
    Image<float>::Ptr depth_uncertainty_ = nullptr;
    Transformation::Ptr motion_predition_ = nullptr;
    Frame::Ptr blur_frame_;
    Frame::Ptr sharp_frame_;
    Camera::Ptr camera_;
    Tracker::Ptr tracker_;
    TrackerCfg options_;
    fs::path config_json_path_;
    std::vector<std::string> vec_im_names_;
};

}  // namespace pymbavo
