/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "IOWrapper/Output3DWrapper.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "core/sensors/CameraPinhole.h"
#include "util/NumType.h"
#include "util/settings.h"
#include "vector"
#include "vo/blur_aware_tracker/blur_aware_direct_tracker.h"
#include <cereal/archives/json.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <math.h>

namespace dso
{
	struct CalibHessian;
	struct FrameHessian;
	struct PointFrameResidual;

	struct Configuration
	{
		std::string dataset_type = "unreal";
		std::string root_path = "/home/trimbot/Desktop/peidong/Unreal4/ArchVizInterior_eth3d_camera_shake_3_trajectory";
		std::string rel_path_to_im_list = "/im_list.txt";
		std::string rel_path_to_ref_ims = "/rgb_sharp_middle/";
		std::string rel_path_to_depth_ims = "/depth_sharp_middle/";
		std::string rel_path_to_curr_ims = "/rgb_blur/";
		std::string rel_path_to_gt_poses = "/groundtruth.txt";
		std::string im_ext = ".png";
		int num_digits_for_im_name = 4;

		int skip_images = 0;
		int num_init_ctrl_knots = 2;
		double dt_frame = 0.0368;
		double exposure_time = 0.0156288;
		double dt_ctrl_knot = 0.03;
		int line_thickness = 2;

		int im_ref_idx = 180;
		int num_curr_imgs = 1;

		int H = 480;
		int W = 768;
		double fx = 548.409;
		double fy = 548.409;
		double cx = 384;
		double cy = 240;

		double extrinsic_Rb2w_qw = 0.5;
		double extrinsic_Rb2w_qx = -0.5;
		double extrinsic_Rb2w_qy = 0.5;
		double extrinsic_Rb2w_qz = -0.5;

		double huber_k = 1;
		std::string patch_pattern_path;
		int num_pyramid_level = 4;
		int offset_curr_img = 1;

		bool init_pose_with_noise = false;

		// chi-square threshold for outlier rejection
		double max_chi_square_error;

		// params for keyframe creation
		double keyframe_max_flow_mag0;
		double keyframe_max_flow_mag1;
		double keyframe_max_flow_mag2;
		double keyframe_max_blur_kernel_mag;

		// params for imshow visualization
		bool with_gui;
		int frame_id_start_to_pause_imshow;

		// linear solver type
		std::string solver_type;
		int num_virtual_images_per_frame[6];

		template <class Archive>
		void serialize(Archive &ar)
		{
			ar(CEREAL_NVP(dataset_type));
			ar(CEREAL_NVP(root_path));
			ar(CEREAL_NVP(rel_path_to_im_list));
			ar(CEREAL_NVP(rel_path_to_ref_ims));
			ar(CEREAL_NVP(rel_path_to_depth_ims));
			ar(CEREAL_NVP(rel_path_to_curr_ims));
			ar(CEREAL_NVP(rel_path_to_gt_poses));
			ar(CEREAL_NVP(im_ext));
			ar(CEREAL_NVP(num_digits_for_im_name));

			ar(CEREAL_NVP(skip_images));
			ar(CEREAL_NVP(num_init_ctrl_knots));
			ar(CEREAL_NVP(dt_frame));
			ar(CEREAL_NVP(exposure_time));
			ar(CEREAL_NVP(dt_ctrl_knot));
			ar(CEREAL_NVP(line_thickness));
			ar(CEREAL_NVP(im_ref_idx));
			ar(CEREAL_NVP(num_curr_imgs));

			ar(CEREAL_NVP(H));
			ar(CEREAL_NVP(W));
			ar(CEREAL_NVP(fx));
			ar(CEREAL_NVP(fy));
			ar(CEREAL_NVP(cx));
			ar(CEREAL_NVP(cy));
			ar(CEREAL_NVP(extrinsic_Rb2w_qw));
			ar(CEREAL_NVP(extrinsic_Rb2w_qx));
			ar(CEREAL_NVP(extrinsic_Rb2w_qy));
			ar(CEREAL_NVP(extrinsic_Rb2w_qz));

			ar(CEREAL_NVP(huber_k));
			ar(CEREAL_NVP(patch_pattern_path));
			ar(CEREAL_NVP(num_pyramid_level));
			ar(CEREAL_NVP(offset_curr_img));

			ar(CEREAL_NVP(init_pose_with_noise));

			ar(CEREAL_NVP(max_chi_square_error));
			ar(CEREAL_NVP(keyframe_max_flow_mag0));
			ar(CEREAL_NVP(keyframe_max_flow_mag1));
			ar(CEREAL_NVP(keyframe_max_flow_mag2));
			ar(CEREAL_NVP(keyframe_max_blur_kernel_mag));
			ar(CEREAL_NVP(with_gui));
			ar(CEREAL_NVP(frame_id_start_to_pause_imshow));
			ar(CEREAL_NVP(solver_type));
			ar(CEREAL_NVP(num_virtual_images_per_frame));
		}

		void make_template(const std::string &filePath)
		{
			Configuration configs;
			std::ofstream ofs(filePath);
			cereal::JSONOutputArchive archive(ofs);
			archive(cereal::make_nvp("Configuration", configs));
		}

		static Configuration load_configs(const std::string &filePath)
		{
			Configuration configs;
			std::ifstream ifs(filePath);
			cereal::JSONInputArchive archive(ifs);
			archive(cereal::make_nvp("Configuration", configs));
			return configs;
		}
	};
	class CoarseTracker
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		CoarseTracker(int w, int h);
		~CoarseTracker();

		bool trackNewestCoarse(
			FrameHessian *newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper *wrap = 0);

		bool trackNewestCoarseBlurAware(
			FrameHessian *newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 &failure_indicators,
			IOWrap::Output3DWrapper *wrap = 0);

		bool relocalizeBlurAware(FrameHessian *newFrameHessian,
								 SE3 &lastToNew_out,
								 Vec5 &failure_indicators);

		void setCoarseTrackingRef(
			std::vector<FrameHessian *> frameHessians);

		void makeK(
			CalibHessian *HCalib);

		bool debugPrint, debugPlot;

		Mat33f K[PYR_LEVELS];
		Mat33f Ki[PYR_LEVELS];
		float fx[PYR_LEVELS];
		float fy[PYR_LEVELS];
		float fxi[PYR_LEVELS];
		float fyi[PYR_LEVELS];
		float cx[PYR_LEVELS];
		float cy[PYR_LEVELS];
		float cxi[PYR_LEVELS];
		float cyi[PYR_LEVELS];
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		void debugPlotIDepthMap(float *minID, float *maxID, std::vector<IOWrap::Output3DWrapper *> &wraps);
		void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps);

		void initializeBlurAwareTracker(std::string &config_path);

		FrameHessian *lastRef;
		AffLight lastRef_aff_g2l;
		FrameHessian *newFrame;
		int refFrameID;

		// act as pure ouptut
		Vec5 lastResiduals;
		Vec3 lastFlowIndicators;
		double firstCoarseRMSE;

	private:
		void makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians);
		float *idepth[PYR_LEVELS];
		float *weightSums[PYR_LEVELS];
		float *weightSums_bak[PYR_LEVELS];

		Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
		void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
		void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);

		void gridSelection(const int num_pts, const float *pts_u, const float *pts_v, const float *pts_iz,
						   const float *dIxy_mag, const int im_H, const int im_W, const int cell_H, const int cell_W, const int min_squared_grad, std::vector<SLAM::Core::Vector2d> &selected_kpts_uv, std::vector<double> &selected_kpts_z);

		// pc buffers
		float *pc_u[PYR_LEVELS];
		float *pc_v[PYR_LEVELS];
		float *pc_idepth[PYR_LEVELS];
		float *pc_color[PYR_LEVELS];
		int pc_n[PYR_LEVELS];

		// warped buffers
		float *buf_warped_idepth;
		float *buf_warped_u;
		float *buf_warped_v;
		float *buf_warped_dx;
		float *buf_warped_dy;
		float *buf_warped_residual;
		float *buf_warped_weight;
		float *buf_warped_refColor;
		int buf_warped_n;

		std::vector<float *> ptrToDelete;

		Accumulator9 acc;

		SLAM::VO::BlurAwareDirectTracker *mBlurAwareTracker;
		SLAM::Core::CameraPinhole *mCamera;
		SLAM::VO::BlurAwareDirectTrackerOptions mOptions;
		Configuration mConfigs;
		std::vector<SLAM::Core::Vector2d> mTmpCpuKptXy[8];
		std::vector<double> mTmpCpuKptZ[8];
	};

	class CoarseDistanceMap
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		CoarseDistanceMap(int w, int h);
		~CoarseDistanceMap();

		void makeDistanceMap(
			std::vector<FrameHessian *> frameHessians,
			FrameHessian *frame);

		void makeInlierVotes(
			std::vector<FrameHessian *> frameHessians);

		void makeK(CalibHessian *HCalib);

		float *fwdWarpedIDDistFinal;

		Mat33f K[PYR_LEVELS];
		Mat33f Ki[PYR_LEVELS];
		float fx[PYR_LEVELS];
		float fy[PYR_LEVELS];
		float fxi[PYR_LEVELS];
		float fyi[PYR_LEVELS];
		float cx[PYR_LEVELS];
		float cy[PYR_LEVELS];
		float cxi[PYR_LEVELS];
		float cyi[PYR_LEVELS];
		int w[PYR_LEVELS];
		int h[PYR_LEVELS];

		void addIntoDistFinal(int u, int v);

	private:
		PointFrameResidual **coarseProjectionGrid;
		int *coarseProjectionGridNum;
		Eigen::Vector2i *bfsList1;
		Eigen::Vector2i *bfsList2;

		void growDistBFS(int bfsNum);
	};

} // namespace dso
