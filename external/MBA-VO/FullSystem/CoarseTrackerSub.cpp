#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "IOWrapper/ImageRW.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "core/measurements/Image.h"
#include "core/sensors/CameraPinhole.h"
#include "utils/Geometry.h"
#include "utils/ImShow.h"
#include "utils/InputOutput.h"
#include "utils/ScalarToColorMap.h"
#include "vo/blur_aware_tracker/blur_aware_direct_tracker.h"
#include <algorithm>
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
    void CoarseTracker::initializeBlurAwareTracker(std::string &path_to_config)
    {
        // std::string path_to_config = "/home/trimbot/Desktop/peidong/dso/thirdparty/slam/test/data/blur_aware_tracker_configs_unreal.json";
        mConfigs = Configuration::load_configs(path_to_config);

        /*
        set up configurations
        */
        Eigen::Vector4d K(mConfigs.fx, mConfigs.fy, mConfigs.cx, mConfigs.cy);
        mCamera = new SLAM::Core::CameraPinhole(K, mConfigs.H, mConfigs.W);

        //
        mOptions.dt_ctrl_knot = mConfigs.dt_ctrl_knot;
        mOptions.spline_deg_k = mConfigs.num_init_ctrl_knots;
        mOptions.dt_frame = mConfigs.dt_frame;
        mOptions.im_size_HW.values[0] = mConfigs.H;
        mOptions.im_size_HW.values[1] = mConfigs.W;
        mOptions.intrinsics.values[0] = mConfigs.fx;
        mOptions.intrinsics.values[1] = mConfigs.fy;
        mOptions.intrinsics.values[2] = mConfigs.cx;
        mOptions.intrinsics.values[3] = mConfigs.cy;

        mOptions.huber_k = mConfigs.huber_k;

        mOptions.num_pyramid_levels = mConfigs.num_pyramid_level;
        mOptions.num_virtual_poses_per_frame[0] = mConfigs.num_virtual_images_per_frame[0];
        mOptions.num_virtual_poses_per_frame[1] = mConfigs.num_virtual_images_per_frame[1];
        mOptions.num_virtual_poses_per_frame[2] = mConfigs.num_virtual_images_per_frame[2];
        mOptions.num_virtual_poses_per_frame[3] = mConfigs.num_virtual_images_per_frame[3];
        mOptions.num_virtual_poses_per_frame[4] = mConfigs.num_virtual_images_per_frame[4];
        mOptions.num_virtual_poses_per_frame[5] = mConfigs.num_virtual_images_per_frame[5];

        mOptions.max_num_frames = 1;
        mOptions.max_num_virtual_sharp_imgs_per_frame = 64;
        mOptions.max_num_keypoints = 500;
        mOptions.max_patch_size = 121;

        mOptions.max_chi_square_error = mConfigs.max_chi_square_error;
        mOptions.keyframe_max_flow_mag0 = mConfigs.keyframe_max_flow_mag0;
        mOptions.keyframe_max_flow_mag1 = mConfigs.keyframe_max_flow_mag1;
        mOptions.keyframe_max_flow_mag2 = mConfigs.keyframe_max_flow_mag2;

        mOptions.keyframe_max_blur_kernel_mag = mConfigs.keyframe_max_blur_kernel_mag;
        mOptions.with_gui = mConfigs.with_gui;
        mOptions.frame_id_start_to_pause_imshow = mConfigs.frame_id_start_to_pause_imshow;
        mOptions.line_thickness = mConfigs.line_thickness;
        mOptions.solver_type = mConfigs.solver_type;

        // load in patch pattern
        std::ifstream reader;
        reader.open(mConfigs.patch_pattern_path, std::ifstream::in);
        if (!reader.is_open())
        {
            std::cerr << "fails to load in patch pattern file " << mConfigs.patch_pattern_path << "\n";
            std::exit(0);
        }
        std::vector<int> pattern;
        while (!reader.eof())
        {
            int x;
            reader >> x;
            pattern.push_back(x);
        }

        mOptions.patch_size[0] = pattern.size() / 2;
        mOptions.patch_size[1] = pattern.size() / 2;
        mOptions.patch_size[2] = pattern.size() / 2;
        mOptions.patch_size[3] = pattern.size() / 2;
        mOptions.patch_size[4] = pattern.size() / 2;
        mOptions.patch_size[5] = pattern.size() / 2;

        for (int lvl = 0; lvl < mOptions.num_pyramid_levels; lvl++)
        {
            mOptions.local_patch_pattern_xy[lvl] = new int[mOptions.patch_size[lvl] * 2];
            for (int i = 0; i < mOptions.patch_size[lvl]; i++)
            {
                mOptions.local_patch_pattern_xy[lvl][2 * i] = pattern[2 * i];
                mOptions.local_patch_pattern_xy[lvl][2 * i + 1] = pattern[2 * i + 1];
            }
        }
        /*
        create blur aware tracker
        */
        mBlurAwareTracker = new SLAM::VO::BlurAwareDirectTracker(mOptions);
        mBlurAwareTracker->setCamera(mCamera);
    }

    void CoarseTracker::gridSelection(const int num_pts, const float *pts_u, const float *pts_v, const float *pts_iz,
                                      const float *dIxy_mag, const int im_H, const int im_W, const int cell_H, const int cell_W, const int min_squared_grad, std::vector<SLAM::Core::Vector2d> &selected_kpts_uv, std::vector<double> &selected_kpts_z)
    {
        if (cell_H == 0 || cell_W == 0)
        {
            selected_kpts_z.clear();
            selected_kpts_uv.clear();

            for (int i = 0; i < num_pts; i++)
            {
                selected_kpts_uv.push_back(SLAM::Core::Vector2d(pts_u[i], pts_v[i]));
                selected_kpts_z.push_back(1. / pts_iz[i]);
            }
            return;
        }

        int nCellsH = im_H / cell_H + 1;
        int nCellsW = im_W / cell_W + 1;

        std::vector<float> selected_kpts_mag(nCellsH * nCellsW, 0);
        std::vector<int> selected_kpts_idx(nCellsH * nCellsW, -1);
        for (int i = 0; i < num_pts; i++)
        {
            int cell_index_H = pts_v[i] / cell_H;
            int cell_index_W = pts_u[i] / cell_W;
            int cell_index = cell_index_H * nCellsW + cell_index_W;

            int pt_im_idx = (int)(pts_v[i]) * im_W + int(pts_u[i]);
            if (dIxy_mag[pt_im_idx] > selected_kpts_mag[cell_index] && dIxy_mag[pt_im_idx] > min_squared_grad)
            {
                selected_kpts_mag[cell_index] = dIxy_mag[pt_im_idx];
                selected_kpts_idx[cell_index] = i;
            }
        }

        for (int i = 0; i < selected_kpts_idx.size(); i++)
        {
            if (selected_kpts_idx[i] == -1)
            {
                continue;
            }

            const int idx = selected_kpts_idx[i];
            selected_kpts_uv.push_back(SLAM::Core::Vector2d(pts_u[idx], pts_v[idx]));
            selected_kpts_z.push_back(1. / pts_iz[idx]);
        }

        if (selected_kpts_z.size() == num_pts)
        {
            return;
        }

        if (selected_kpts_z.size() < 50)
        {
            selected_kpts_z.clear();
            selected_kpts_uv.clear();
            gridSelection(num_pts, pts_u, pts_v, pts_iz, dIxy_mag, im_H, im_W, cell_H * 0.95, cell_W * 0.95, min_squared_grad * 0.6, selected_kpts_uv, selected_kpts_z);
        }
    }

    bool CoarseTracker::trackNewestCoarseBlurAware(
        FrameHessian *newFrameHessian,
        SE3 &lastToNew_out, AffLight &aff_g2l_out,
        int coarsestLvl,
        Vec5 &failure_indicators,
        IOWrap::Output3DWrapper *wrap)
    {
        assert(coarsestLvl == mOptions.num_pyramid_levels - 1);

        SLAM::Core::Frame *internal_keyframe = mBlurAwareTracker->getKeyframe();
        if (internal_keyframe == nullptr || internal_keyframe->getFrameId() != lastRef->shell->incoming_id)
        {
            // free internal keyframe
            delete internal_keyframe;

            // create new keyframe & insert to blur aware tracker
            Eigen::Vector3f *image = lastRef->dIp[0];
            SLAM::Core::Image<unsigned char> image_ref(h[0], w[0], 1);
            unsigned char *image_ref_ptr = image_ref.getData();
            for (int r = 0; r < h[0]; r++)
            {
                for (int c = 0; c < w[0]; c++, image_ref_ptr++, image++)
                {
                    *image_ref_ptr = (*image)(0);
                }
            }

            SLAM::Core::Frame *keyframe = new SLAM::Core::Frame(lastRef->shell->incoming_id);
            keyframe->copyImageFrom(mOptions.camera_id,
                                    image_ref.getData(),
                                    image_ref.nHeight(),
                                    image_ref.nWidth(),
                                    1);

            keyframe->setCaptureTime(lastRef->shell->timestamp);
            keyframe->setExposureTime(mConfigs.exposure_time);

            // process keyframe
            keyframe->computeImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);
            keyframe->computeGradImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);

            for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
            {
                keyframe->getImagePtr(mOptions.camera_id, lv)->uploadToGpu();
                keyframe->getGradImagePtr(mOptions.camera_id, lv)->uploadToGpu();
            }

            // // detect keypoints
            // SLAM::Core::FeatureDetectorOptions feat_options;
            // feat_options.detector_type = ENUM_SEMIDENSE;
            // feat_options.score_threshold = 20;
            // feat_options.grid_selection_cell_H = 30;
            // feat_options.grid_selection_cell_W = 30;
            // keyframe->detectFeatures(mOptions.camera_id, feat_options);

            // // assign depth
            // for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
            // {
            //     double scale = pow(2, lv);
            //     std::vector<cv::KeyPoint> &cvKeypoints = keyframe->getFeaturePoints(mOptions.camera_id, lv);

            //     mTmpCpuKptXy[lv].clear();
            //     mTmpCpuKptZ[lv].clear();

            //     int nl = pc_n[lv];
            //     float *lpc_u = pc_u[lv];
            //     float *lpc_v = pc_v[lv];
            //     float *lpc_idepth = pc_idepth[lv];

            //     for (auto &kpt : cvKeypoints)
            //     {
            //         double iz;
            //         double min_dist2 = 1e6;
            //         for (int i = 0; i < nl; i++)
            //         {
            //             float u = lpc_u[i];
            //             float v = lpc_v[i];

            //             float du = u - kpt.pt.x;
            //             float dv = v - kpt.pt.y;

            //             if (du * du + dv * dv < min_dist2)
            //             {
            //                 iz = lpc_idepth[i];
            //                 min_dist2 = du * du + dv * dv;
            //             }
            //         }

            //         if (min_dist2 > 16)
            //         {
            //             continue;
            //         }

            //         mTmpCpuKptXy[lv].push_back(SLAM::Core::Vector2d(kpt.pt.x, kpt.pt.y));
            //         mTmpCpuKptZ[lv].push_back(1. / iz);
            //     }

            //     SLAM::VO::BlurAwareDirectTrackerOptions &options = mBlurAwareTracker->getOptions();
            //     options.num_keypoints[lv] = mTmpCpuKptXy[lv].size();
            //     options.tmp_keypoints_xy[lv] = mTmpCpuKptXy[lv].data();
            //     options.tmp_keypoints_z[lv] = mTmpCpuKptZ[lv].data();
            // }

            const int min_squared_grad = 10 * 10;
            const int cell_H0 = 30;
            const int cell_W0 = 30;

            for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
            {
                int nl = pc_n[lv];
                float *lpc_u = pc_u[lv];
                float *lpc_v = pc_v[lv];
                float *lpc_idepth = pc_idepth[lv];

                const int cell_H_lv = cell_H0 / std::pow(1.414, lv);
                const int cell_W_lv = cell_W0 / std::pow(1.414, lv);

                mTmpCpuKptXy[lv].clear();
                mTmpCpuKptZ[lv].clear();

                gridSelection(nl, lpc_u, lpc_v, lpc_idepth, lastRef->absSquaredGrad[lv], hG[lv], wG[lv], cell_H_lv, cell_W_lv, min_squared_grad, mTmpCpuKptXy[lv], mTmpCpuKptZ[lv]);

                SLAM::VO::BlurAwareDirectTrackerOptions &options = mBlurAwareTracker->getOptions();
                options.num_keypoints[lv] = mTmpCpuKptXy[lv].size();
                options.tmp_keypoints_xy[lv] = mTmpCpuKptXy[lv].data();
                options.tmp_keypoints_z[lv] = mTmpCpuKptZ[lv].data();
            }
            // insert keyframe to tracker
            mBlurAwareTracker->setKeyframe(keyframe);
        }

        // set up current frame
        Eigen::Vector3f *image = newFrameHessian->dIp[0];
        SLAM::Core::Image<unsigned char> image_cur(h[0], w[0], 1);
        unsigned char *image_cur_ptr = image_cur.getData();
        for (int r = 0; r < h[0]; r++)
        {
            for (int c = 0; c < w[0]; c++, image_cur_ptr++, image++)
            {
                *image_cur_ptr = (*image)(0);
            }
        }

        SLAM::Core::Frame *curframe = new SLAM::Core::Frame(newFrameHessian->shell->incoming_id);
        curframe->copyImageFrom(mOptions.camera_id,
                                image_cur.getData(),
                                image_cur.nHeight(),
                                image_cur.nWidth(),
                                1);

        curframe->setCaptureTime(newFrameHessian->shell->timestamp);
        curframe->setExposureTime(mConfigs.exposure_time);

        curframe->computeImagePyramid(mOptions.camera_id, mOptions.num_pyramid_levels);
        for (int lv = 0; lv < mOptions.num_pyramid_levels; lv++)
        {
            curframe->getImagePtr(mOptions.camera_id, lv)->uploadToGpu();
        }

        mBlurAwareTracker->insertCurrentFrame(curframe);

        // initialize spline trajectory
        SLAM::Core::SplineSE3 *splineTrajectory = mBlurAwareTracker->getSplineTrajectory();
        if (splineTrajectory->get_num_knots() == 0)
        {
            splineTrajectory->setSplineDegK(mOptions.spline_deg_k);
            splineTrajectory->InsertControlKnot(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
            splineTrajectory->InsertControlKnot(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero());
        }

        SE3 T_cur2keyframe = lastToNew_out.inverse();
        Eigen::Quaterniond R_b2ref = T_cur2keyframe.unit_quaternion();

        Eigen::Vector3d t_b2ref = T_cur2keyframe.translation();
        splineTrajectory->setStartTime(curframe->getCaptureTime());
        splineTrajectory->TransformTo(curframe->getCaptureTime(), R_b2ref, t_b2ref);

        // do tracking
        mBlurAwareTracker->optimizeTrajectory();

        Eigen::Quaterniond R_cur2ref;
        Eigen::Vector3d t_cur2ref;
        splineTrajectory->GetPose(curframe->getCaptureTime(), R_cur2ref, t_cur2ref);

        //
        SE3 curToRef(R_cur2ref, t_cur2ref);
        lastToNew_out = curToRef.inverse();

        //
        double avg_t_flow_mag = 0;
        double avg_Rt_flow_mag = 0;
        double avg_blur_kernel = 0;
        bool is_keyframe = mBlurAwareTracker->isKeyframe(mCamera, avg_t_flow_mag, avg_Rt_flow_mag, avg_blur_kernel);

        failure_indicators[0] = avg_Rt_flow_mag;
        failure_indicators[1] = avg_blur_kernel;
        failure_indicators[2] = avg_t_flow_mag;

        return is_keyframe;
    }

    bool CoarseTracker::relocalizeBlurAware(FrameHessian *newFrameHessian, SE3 &lastToNew_out, Vec5 &failure_indicators)
    {
        SE3 T_cur2keyframe = lastToNew_out.inverse();
        Eigen::Quaterniond R_b2ref = T_cur2keyframe.unit_quaternion();

        Eigen::Vector3d t_b2ref = T_cur2keyframe.translation();

        SLAM::Core::SplineSE3 *splineTrajectory = mBlurAwareTracker->getSplineTrajectory();
        splineTrajectory->getCtrlKnotsR()[0].setIdentity();
        splineTrajectory->getCtrlKnotsR()[1].setIdentity();
        splineTrajectory->getCtrlKnotst()[0].setZero();
        splineTrajectory->getCtrlKnotst()[1].setZero();
        splineTrajectory->TransformTo(newFrameHessian->shell->timestamp, R_b2ref, t_b2ref);

        // do tracking
        mBlurAwareTracker->optimizeTrajectory();

        Eigen::Quaterniond R_cur2ref;
        Eigen::Vector3d t_cur2ref;
        splineTrajectory->GetPose(newFrameHessian->shell->timestamp, R_cur2ref, t_cur2ref);

        //
        SE3 curToRef(R_cur2ref, t_cur2ref);
        lastToNew_out = curToRef.inverse();

        double avg_Rt_flow_mag = 0;
        double avg_blur_kernel = 0;
        double avg_t_flow_mag = 0;
        bool is_keyframe = mBlurAwareTracker->isKeyframe(mCamera, avg_t_flow_mag, avg_Rt_flow_mag, avg_blur_kernel);

        std::cout << "Is keyframe " << is_keyframe << "\n";

        failure_indicators[0] = avg_Rt_flow_mag;
        failure_indicators[1] = avg_blur_kernel;
        failure_indicators[2] = avg_t_flow_mag;

        return is_keyframe;
    }
} // namespace dso