#ifndef SLAM_VO_BLUR_AWARE_DIRECT_TRACKER_H
#define SLAM_VO_BLUR_AWARE_DIRECT_TRACKER_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "BA_tracker/core/common/Spline.h"
#include "BA_tracker/core/measurements/Frame.h"
#include "failure_indicators.h"
#include "levenberg_marquardt_strategy.h"
#include "spline_update_step.h"
#include "trust_region_step_evaluator.h"

namespace Eigen {
    using Vector5d = ::Eigen::Matrix<double, 5, 1>;
    using Vector6d = ::Eigen::Matrix<double, 6, 1>;
}

namespace SLAM {
namespace VO {

struct BlurAwareDirectTrackerOptions {
    int num_keypoints[8];
    Core::Vector2d *tmp_keypoints_xy[8];
    double *tmp_keypoints_z[8];
    // storage used in pybind
    std::shared_ptr<std::vector<Core::Vector2d>> tmp_keypoints_xy_pybind[8];
    std::shared_ptr<Eigen::VectorXd> tmp_keypoints_z_pybind[8];

    //
    int camera_id;
    Core::VectorX<double, 4> intrinsics;
    Core::VectorX<int, 2> im_size_HW;

    //
    int num_pyramid_levels;
    int num_virtual_poses_per_frame[8];
    int patch_size[8];
    int *local_patch_pattern_xy[8];

    // detector options
    float score_threshold = 25;
    int grid_selection_cell_H = 30;
    int grid_selection_cell_W = 30;
    int max_num_features_to_detect = 1000;

    // huber robust function
    double huber_k = 0.1;

    // solver options
    int max_consecutive_nonmonotonic_steps = 5;
    int max_num_iterations = 50;
    double min_step_quality = 0.5;
    double min_abs_cost_decrease = 0.001;
    std::string solver_type = "SVD_JACOBI";  // "LDLT"

    // options for shared storages
    int max_num_frames = 16;
    int max_num_virtual_sharp_imgs_per_frame = 64;
    int max_num_keypoints = 1024;
    int max_patch_size = 128;
    int max_num_ctrl_knots = 16;

    // options for spline trajectory
    int spline_deg_k = 2;
    double dt_frame;
    double dt_ctrl_knot;

    // chi-square threshold for outlier rejection
    double max_chi_square_error;

    // params for keyframe creation
    double keyframe_max_flow_mag0;
    double keyframe_max_flow_mag1;
    double keyframe_max_flow_mag2;
    double keyframe_max_blur_kernel_mag;

    // params for imshow visualization
    bool with_gui = false;
    int frame_id_start_to_pause_imshow = 0;
    int line_thickness = 1;
};

struct SolverSummary {
    int num_iteration = 0;
    double abs_cost_decrease = 1e10;
};

class BlurAwareDirectTracker {
   public:
    using Ptr = std::shared_ptr<BlurAwareDirectTracker>;
    BlurAwareDirectTracker(const BlurAwareDirectTrackerOptions& options);
    ~BlurAwareDirectTracker();

   public:
    void setKeyframe(Core::Frame::Ptr keyframe);
    void setCurrentFrames(const std::vector<Core::Frame::Ptr> &currentFrames);
    void insertCurrentFrame(Core::Frame::Ptr frame);
    void setCamera(Core::CameraBase::Ptr camera);

   public:
    Core::Frame::Ptr getKeyframe();
    BlurAwareDirectTrackerOptions &getOptions();
    double getFinalEnergy();

   public:
    Core::SplineSE3::Ptr getSplineTrajectory();

   public:
    bool track(Core::Frame::Ptr blur_frame);
    std::tuple<bool, FailureIndicators> trackCoarse(
        Core::Frame::Ptr blur_frame,
        Core::Transformation::Ptr motion_prediction);

    std::tuple<Core::Transformation, Core::Transformation, Core::Transformation>
    evalPoses(bool is_keyframe, Core::Frame::Ptr blur_frame);

    void optimizeTrajectory();

   public:
    std::tuple<bool, FailureIndicators> isKeyframe();

   private:
    void optimizePyramidLevel(int pyra_level);
    void uploadDataToGpu();
    void uploadDataToGpu(int pyra_level);
    void evaluateCostGradientAndHessian(int pyra_level);
    bool computeTrustRegionStep();
    void computeCandidatePointAndEvaluateCost(int pyra_level);
    void handleInvalidStep();
    bool isStepSuccessful();
    void handleSuccessfulStep(int pyra_level);
    void handleUnsuccessfulStep();
    bool finalizeIterationAndCheckIfMinimizerCanContinue(SolverSummary &summary);
    void detectOutliersAndUploadToGpu(std::vector<uchar> &outlier_flags);

   public:
    void tmpProcessKeyframe(Core::Frame::Ptr keyframe,
                            Core::Image<float>::Ptr depth_image,
                            Core::Image<float>::Ptr depth_uncertainty);
    void drawKeyframeKeypoints(int pyra_level);
    void drawCurrFrameKeypoints(int pyrd_lv,
                                Core::CameraBase::Ptr camera,
                                const std::string &winText);
    void drawErrorResiduals(int pyrd_lv, std::vector<uchar> &outlier_flags);
    Core::FeatureDetectorOptions::Ptr getDetectorOptions() const;

   public:
    // expose temp storages
    BlurAwareDirectTrackerOptions mOptions;

   public:
    double getStepQuality() const {return mStepQuality;}

   private:
    Core::Frame::Ptr mKeyframe;
    std::vector<Core::Frame::Ptr> mCurrentFrames;
    Core::SplineSE3::Ptr mSplineTrajectory;
    CudaSharedStorages mCudaSharedStorages;

    LevenbergMarquardtStrategy *mLmStrategy;
    TrustRegionStepEvaluator *mStepEvaluator;
    double mEvaluationPointCost{};
    double mCandidatePointCost{};
    double mQuadraticApproximatedCost{};
    double mStepQuality{};

    Eigen::MatrixXd mHessian;
    Eigen::VectorXd mGradient;
    Eigen::VectorXd mTrustRegionStep;
    double *mCandidatePoint_t{};
    double *mCandidatePoint_R{};

    std::vector<int> mFramesCtrlKnotSegStartIndices;

    //
    bool mIsFirstFrame;
    double mPrevTimestamp{};
    Core::Transformation mTprevB2W;
    Core::Transformation mdTneighFrames;
    Core::Transformation mdTspline;
    Eigen::Vector6d mSplineVelocity;
    Eigen::Vector6d mNeighFrameVelocity;

    double mAvgKernelLength;

    //
    std::vector<Core::Vector2d> mTmpCpuKptXy[8];
    std::vector<double> mTmpCpuKptZ[8];
    Core::CameraBase::Ptr mCamera;

    Core::FeatureDetectorOptions::Ptr mFeatureDetectorOptions;

    //
    Core::Transformation mTKeyframe;
};
}  // namespace VO
}  // namespace SLAM

#endif