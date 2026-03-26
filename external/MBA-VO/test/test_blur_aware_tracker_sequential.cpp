#include "BA_tracker/core/measurements/Image.h"
#include "BA_tracker/core/sensors/CameraPinhole.h"
#include "BA_tracker/utils/Configuration.h"
#include "BA_tracker/utils/Geometry.h"
#include "BA_tracker/utils/ImShow.h"
#include "BA_tracker/utils/InputOutput.h"
#include "BA_tracker/ba_tracker/blur_aware_direct_tracker.h"
#include "BA_tracker/ba_tracker/generate_synthetic_data.h"
#include <Eigen/Dense>
#include <filesystem>
#include <iostream>
#include <string>

using namespace SLAM::VO;
using namespace SLAM::Core;
using namespace SLAM::Utils;
namespace fs = std::filesystem;

void load_ground_truth(fs::path& path_to_file, std::vector<std::array<double, 8>> &gt_poses)
{
    std::ifstream fileReader;
    fileReader.open(path_to_file, std::ifstream::in);
    if (!fileReader.is_open())
    {
        printf("%s fails to load %s, quit...\n", __FUNCTION__, path_to_file.c_str());
        std::exit(0);
    }

    while (!fileReader.eof())
    {
        std::string line;
        std::getline(fileReader, line);
        if (line.find('#') != std::string::npos || line.empty())
            continue;

        std::stringstream line_stream(line);

        std::array<double, 8> pose;
        line_stream >> pose[0] >> pose[1] >> pose[2] >> pose[3] >> pose[4] >> pose[5] >> pose[6] >> pose[7];
        gt_poses.push_back(pose);
    }
}

std::array<double, 8> get_pose(const std::vector<std::array<double, 8>> &gt_poses, double t)
{
    for (auto &pose : gt_poses)
    {
        if (std::fabs(pose[0] - t) < 0.0099)
        {
            return pose;
        }
    }
    assert(false);
}

void load_image_list(fs::path& path_to_list, int num_digits_for_im_name, std::vector<std::string> &vec_im_names)
{
    std::ifstream fileReader;
    fileReader.open(path_to_list, std::ifstream::in);
    if (!fileReader.is_open())
    {
        printf("%s fails to load %s, quit...\n", __FUNCTION__, path_to_list.c_str());
        std::exit(0);
    }

    while (!fileReader.eof())
    {
        std::string line;
        std::getline(fileReader, line);
        if (line.find('#') != std::string::npos || line.empty())
            continue;

        line = line.substr(0, line.find(" "));

        // check if we need to do conversion
        int digit_pos = line.find(".");
        int num_real_digits = line.length() - digit_pos - 1;

        if (num_real_digits != num_digits_for_im_name)
        {
            int num_chars_to_remove = num_real_digits - num_digits_for_im_name;
            line = line.substr(0, line.length() - num_chars_to_remove);
        }
        vec_im_names.push_back(line);
    }
}

void test_tracker_sequential(fs::path path_to_config)
{
    std::cout << "\n------------------ " << __FUNCTION__ << "------------------\n";
    Configuration configs = Configuration::load_configs(path_to_config);

    // load in image file names
    fs::path path_to_im_list = configs.root_path / configs.rel_path_to_im_list;
    std::vector<std::string> vec_im_names;
    load_image_list(path_to_im_list, configs.num_digits_for_im_name, vec_im_names);

    /*
     set up configurations
     */
    Eigen::Vector4d K(configs.fx, configs.fy, configs.cx, configs.cy);
    auto camera = std::make_shared<CameraPinhole>(K, configs.H, configs.W);

    //
    BlurAwareDirectTrackerOptions options;
    options.dt_ctrl_knot = configs.dt_ctrl_knot;
    options.spline_deg_k = configs.num_init_ctrl_knots;
    options.dt_frame = configs.dt_frame;
    options.im_size_HW.values[0] = configs.H;
    options.im_size_HW.values[1] = configs.W;
    options.intrinsics.values[0] = configs.fx;
    options.intrinsics.values[1] = configs.fy;
    options.intrinsics.values[2] = configs.cx;
    options.intrinsics.values[3] = configs.cy;

    options.huber_k = configs.huber_k;

    options.num_pyramid_levels = configs.num_pyramid_level;
    options.num_virtual_poses_per_frame[0] = configs.num_virtual_images_per_frame[0];
    options.num_virtual_poses_per_frame[1] = configs.num_virtual_images_per_frame[1];
    options.num_virtual_poses_per_frame[2] = configs.num_virtual_images_per_frame[2];
    options.num_virtual_poses_per_frame[3] = configs.num_virtual_images_per_frame[3];
    options.num_virtual_poses_per_frame[4] = configs.num_virtual_images_per_frame[4];
    options.num_virtual_poses_per_frame[5] = configs.num_virtual_images_per_frame[5];

    options.max_num_frames = 1;
    options.max_num_virtual_sharp_imgs_per_frame = 64;
    options.max_num_keypoints = 500;
    options.max_patch_size = 121;

    options.max_chi_square_error = configs.max_chi_square_error;
    options.keyframe_max_flow_mag0 = configs.keyframe_max_flow_mag0;
    options.keyframe_max_flow_mag1 = configs.keyframe_max_flow_mag1;
    options.keyframe_max_blur_kernel_mag = configs.keyframe_max_blur_kernel_mag;

    options.with_gui = configs.with_gui;
    options.frame_id_start_to_pause_imshow = configs.frame_id_start_to_pause_imshow;
    options.line_thickness = configs.line_thickness;

    options.solver_type = configs.solver_type;

    // load in patch pattern
    std::ifstream reader;
    reader.open(configs.patch_pattern_path, std::ifstream::in);
    if (!reader.is_open())
    {
        std::cerr << "fails to load in patch pattern file " << configs.patch_pattern_path << "\n";
        std::exit(0);
    }
    std::vector<int> pattern;
    while (!reader.eof())
    {
        int x;
        reader >> x;
        pattern.push_back(x);
    }

    options.patch_size[0] = pattern.size() / 2;
    options.patch_size[1] = pattern.size() / 2;
    options.patch_size[2] = pattern.size() / 2;
    options.patch_size[3] = pattern.size() / 2;
    options.patch_size[4] = pattern.size() / 2;
    options.patch_size[5] = pattern.size() / 2;

    for (int lvl = 0; lvl < options.num_pyramid_levels; lvl++)
    {
        options.local_patch_pattern_xy[lvl] = new int[options.patch_size[lvl] * 2];
        for (int i = 0; i < options.patch_size[lvl]; i++)
        {
            options.local_patch_pattern_xy[lvl][2 * i] = pattern[2 * i];
            options.local_patch_pattern_xy[lvl][2 * i + 1] = pattern[2 * i + 1];
        }
    }
    /*
     create blur aware tracker
     */
    BlurAwareDirectTracker tracker(options);
    tracker.setCamera(camera);

    /**
     *  initialize spline trajectory
    **/
    if (!configs.init_pose_with_noise)
    {
        Eigen::Quaterniond Rc2b(configs.extrinsic_Rb2w_qw,
                                configs.extrinsic_Rb2w_qx,
                                configs.extrinsic_Rb2w_qy,
                                configs.extrinsic_Rb2w_qz);

        fs::path ground_truth_pose_path = configs.root_path / configs.rel_path_to_gt_poses;
        std::vector<std::array<double, 8>> gt_poses;
        load_ground_truth(ground_truth_pose_path, gt_poses);

        double spline_start_t0 = (long long)((std::stof(vec_im_names[configs.im_ref_idx + 1]) - 0.5 * configs.exposure_time) * 100) * 0.01;
        tracker.getSplineTrajectory()->setSplineDegK(options.spline_deg_k);
        tracker.getSplineTrajectory()->setSamplingInterval(options.dt_ctrl_knot);
        tracker.getSplineTrajectory()->setStartTime(spline_start_t0);

        for (int i = 0; i < configs.num_init_ctrl_knots; i++)
        {
            double t = spline_start_t0 + i * options.dt_ctrl_knot;
            std::array<double, 8> pose = get_pose(gt_poses, t);
            // printf("t %f, %f %f %f %f %f %f %f\n",
            //        pose[0],
            //        pose[1],
            //        pose[2],
            //        pose[3],
            //        pose[4],
            //        pose[5],
            //        pose[6],
            //        pose[7]);
            Eigen::Vector3d ti(pose[1], pose[2], pose[3]);
            Eigen::Quaterniond Ri(pose[7], pose[4], pose[5], pose[6]);
            Ri = Ri * Rc2b;
            tracker.getSplineTrajectory()->InsertControlKnot(Ri, ti);
        }

        std::array<double, 8> pose = get_pose(gt_poses, std::stof(vec_im_names[configs.im_ref_idx]));
        Eigen::Vector3d t_ref2w(pose[1], pose[2], pose[3]);
        Eigen::Quaterniond R_ref2w(pose[7], pose[4], pose[5], pose[6]);
        R_ref2w = R_ref2w * Rc2b;
        Transformation T_ref2w(R_ref2w, t_ref2w);
        Transformation dT = T_ref2w.inverse();
        tracker.getSplineTrajectory()->TransformBy(dT.getRotation(), dT.getTranslation());
    }

    /*
     load in frames
     */
    std::ofstream estimatedTrajFile;
    estimatedTrajFile.open("results.txt");
    estimatedTrajFile << std::setprecision(15);

    SLAM::Core::Transformation T_last_kf;
    for (int i = configs.im_ref_idx; i < vec_im_names.size(); i++)
    {
        std::cout << "\nLoad in image: " << configs.root_path / configs.rel_path_to_ref_ims / (vec_im_names[i] + configs.im_ext) << "\n";

        std::cout << "Track frame: " << i << "\n";

        // load in reference frame
        cv::Mat I_sharp = cv::imread(configs.root_path / configs.rel_path_to_ref_ims / (vec_im_names[i] + configs.im_ext), 0);

        auto sharp_frame = std::make_shared<Frame>(i);
        sharp_frame->copyImageFrom(options.camera_id, I_sharp.data, I_sharp.rows, I_sharp.cols, I_sharp.channels());
        sharp_frame->setCaptureTime(std::stof(vec_im_names[i]));

        // load in blur frame
        cv::Mat I_blur = cv::imread(configs.root_path / configs.rel_path_to_curr_ims / (vec_im_names[i+1] + configs.im_ext), 0);

        auto blur_frame = std::make_shared<Frame>(i);
        blur_frame->copyImageFrom(options.camera_id, I_blur.data, I_blur.rows, I_blur.cols, I_blur.channels());
        blur_frame->setCaptureTime(std::stof(vec_im_names[i+1]));
        blur_frame->setExposureTime(configs.exposure_time);

        //
        std::string path_to_depth;
        auto depth_image = std::make_shared<Image<float>>(configs.H, configs.W, 1);
        if ("unreal" == configs.dataset_type)
        {
            // path_to_depth = configs.root_path / configs.rel_path_to_depth_ims / (vec_im_names[i] + ".exr");
            // auto depth_ray_d = std::make_shared<Image<float>>(configs.H, configs.W, 1);
            // load_depthMap(path_to_depth, depth_ray_d);
            // convert_ray_d_to_z(camera, depth_ray_d, depth_image);
            path_to_depth = configs.root_path / "depth_exr_middle" / (vec_im_names[i] + ".exr");
            cv::Mat depth = cv::imread(path_to_depth, cv::IMREAD_ANYDEPTH);
            cv::Mat depthf;
            depth.convertTo(depthf, CV_32FC1);
            depth_image->copyFrom((float *)depthf.data, configs.H, configs.W, 1);
        }
        else if ("eth3d" == configs.dataset_type)
        {
            path_to_depth = configs.root_path / configs.rel_path_to_depth_ims / (vec_im_names[i] + ".png");
            cv::Mat depth = cv::imread(path_to_depth, cv::IMREAD_ANYDEPTH);
            cv::Mat depthf;
            depth.convertTo(depthf, CV_32FC1);
            depthf = depthf / 5000;
            depth_image->copyFrom((float *)depthf.data, configs.H, configs.W, 1);
        }

        if (i == configs.im_ref_idx)
        {
            tracker.setKeyframe(sharp_frame);
            tracker.tmpProcessKeyframe(sharp_frame, depth_image, nullptr);
        }

        bool isKeyframe = tracker.track(blur_frame);
        if (isKeyframe)
        {
            tracker.setKeyframe(sharp_frame);
            tracker.tmpProcessKeyframe(sharp_frame, depth_image, nullptr);
        }
        auto [T_start, T_b2w, T_end] = tracker.evalPoses(isKeyframe, blur_frame);

        T_b2w = T_last_kf * T_b2w;
        if (isKeyframe)
        {
            T_last_kf = T_b2w;
        }

        estimatedTrajFile << blur_frame->getCaptureTime() << " "
                          << static_cast<Transformation>(T_b2w).getTranslation()(0) << " "
                          << static_cast<Transformation>(T_b2w).getTranslation()(1) << " "
                          << static_cast<Transformation>(T_b2w).getTranslation()(2) << " "
                          << static_cast<Transformation>(T_b2w).getRotation().x() << " "
                          << static_cast<Transformation>(T_b2w).getRotation().y() << " "
                          << static_cast<Transformation>(T_b2w).getRotation().z() << " "
                          << static_cast<Transformation>(T_b2w).getRotation().w() << "\n";
    }
    estimatedTrajFile.close();
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: ./test_blur_aware_tracker_sequential path_to_config\n";
        std::exit(0);
    }
    test_tracker_sequential(fs::path(argv[1]));
    return 0;
}