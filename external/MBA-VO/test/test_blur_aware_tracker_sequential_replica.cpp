#include <set>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "BA_tracker/core/measurements/Image.h"
#include "BA_tracker/core/sensors/CameraPinhole.h"
#include "BA_tracker/utils/Configuration.h"
#include "BA_tracker/utils/Geometry.h"
#include "BA_tracker/utils/ImShow.h"
#include "BA_tracker/utils/InputOutput.h"
#include "BA_tracker/ba_tracker/blur_aware_direct_tracker.h"
#include "BA_tracker/ba_tracker/generate_synthetic_data.h"


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
        if (std::fabs(pose[0] - t) < 0.1)
        {
            return pose;
        }
    }
    assert(false);
}

void load_images(fs::path& pathToSequence, std::vector<fs::path>& vecPathToRgb, std::vector<fs::path>& vecPathToDepth)
{
    if (!fs::exists(pathToSequence))
    {
        std::cerr << "ERROR: Path to sequence does not exist!" << std::endl;
        exit(EXIT_FAILURE);
    }
    //--- filenames are unique so we can use a set
    std::set<fs::path> pathToRgbSorted;
    std::set<fs::path> pathToDepthSorted;
    for (const auto& entry : fs::directory_iterator(pathToSequence))
    {
        if (entry.path().extension() == ".jpg")
        {
            pathToRgbSorted.insert(entry.path());
        }
        else if (entry.path().extension() == ".png")
        {
            pathToDepthSorted.insert(entry.path());
        }
    }

    // copy to vector
    vecPathToRgb.assign(pathToRgbSorted.begin(), pathToRgbSorted.end());
    vecPathToDepth.assign(pathToDepthSorted.begin(), pathToDepthSorted.end());
}

void test_tracker_sequential(fs::path path_to_config)
{
    std::cout << "\n------------------ " << __FUNCTION__ << "------------------\n";
    Configuration configs = Configuration::load_configs(path_to_config);

    // load images
    fs::path pathToImages = configs.root_path / configs.rel_path_to_ref_ims;
    std::vector<fs::path> vecPathToRgb;
    std::vector<fs::path> vecPathToDepth;
    load_images(pathToImages, vecPathToRgb, vecPathToDepth);
    const int nImages = vecPathToRgb.size();

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

        std::string timestamp_im_ref_next = vecPathToRgb[configs.im_ref_idx + 1].filename().stem().string();
        timestamp_im_ref_next = timestamp_im_ref_next.substr(5);
        // std::cout << "timestamp_im_ref_next: " << timestamp_im_ref_next << std::endl;

        double spline_start_t0 = (long long)((std::stod(timestamp_im_ref_next) - 0.5 * configs.exposure_time) * 100) * 0.01;
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
        std::string timestamp_im_ref = vecPathToRgb[configs.im_ref_idx].filename().stem().string();
        timestamp_im_ref = timestamp_im_ref.substr(5);
        // std::cout << "timestamp_im_ref: " << timestamp_im_ref << std::endl;

        std::array<double, 8> pose = get_pose(gt_poses, std::stod(timestamp_im_ref));
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
    bool isKeyframe = true;
    Frame::Ptr sharp_frame;
    auto depth_image = std::make_shared<Image<float>>(configs.H, configs.W, 1);
    for (int i = configs.im_ref_idx; i < nImages - 1; i++)
    {
        const fs::path& pathToRef   = vecPathToRgb[i];
        const fs::path& pathToDepth = vecPathToDepth[i];
        const fs::path& pathToCurr  = vecPathToRgb[i];
        std::cout << "Load in Ref:   " << pathToRef << std::endl;
        std::cout << "Load in Depth: " << pathToDepth << std::endl;
        std::cout << "Load in Curr:  " << pathToCurr << std::endl;
        std::cout << "Tracking frame: " << i << std::endl;

        if (isKeyframe)
        {
            // load in reference frame
            cv::Mat I_sharp = cv::imread(pathToRef.string(), 0);

            sharp_frame = std::make_shared<Frame>(i);
            sharp_frame->copyImageFrom(options.camera_id, I_sharp.data, I_sharp.rows, I_sharp.cols, I_sharp.channels());
            std::string timestamp = vecPathToRgb[i].filename().stem().string();
            timestamp = timestamp.substr(5);
            // std::cout << "sharp_frame timestamp: " << timestamp << std::endl;
            sharp_frame->setCaptureTime(std::stod(timestamp));

            // load in depth of reference frame
            cv::Mat depth = cv::imread(pathToDepth, cv::IMREAD_ANYDEPTH);
            cv::Mat depthf;
            depth.convertTo(depthf, CV_32FC1);
            depth_image->copyFrom((float *)depthf.data, configs.H, configs.W, 1);
        }

        // load in blur frame
        cv::Mat I_blur = cv::imread(pathToCurr.string(), 0);
        auto blur_frame = std::make_shared<Frame>(i);
        blur_frame->copyImageFrom(options.camera_id, I_blur.data, I_blur.rows, I_blur.cols, I_blur.channels());
        std::string timestamp2 = vecPathToRgb[i].filename().stem().string();
        timestamp2 = timestamp2.substr(5);
        // std::cout << "blur_frame timestamp: " << timestamp2 << std::endl;
        blur_frame->setCaptureTime(std::stod(timestamp2));
        blur_frame->setExposureTime(configs.exposure_time);

        if (i == configs.im_ref_idx)
        {
            tracker.setKeyframe(sharp_frame);
            tracker.tmpProcessKeyframe(sharp_frame, depth_image, nullptr);
            continue;
        }
        isKeyframe = tracker.track(blur_frame);

        if (isKeyframe)
        {
            tracker.setKeyframe(sharp_frame);
            tracker.tmpProcessKeyframe(sharp_frame, depth_image, nullptr);
        }

//        if (i % 5 == 0)
//        {
//            isKeyframe = true;
//        }

        auto [T_start, T_b2w, T_end] = tracker.evalPoses(isKeyframe, blur_frame);

        T_b2w = T_last_kf * T_b2w;
        if (isKeyframe)
        {
            T_last_kf = T_b2w;
        }

        estimatedTrajFile << blur_frame->getCaptureTime() << " "
                          << T_b2w.getTranslation()(0) << " "
                          << T_b2w.getTranslation()(1) << " "
                          << T_b2w.getTranslation()(2) << " "
                          << T_b2w.getRotation().x() << " "
                          << T_b2w.getRotation().y() << " "
                          << T_b2w.getRotation().z() << " "
                          << T_b2w.getRotation().w() << "\n";
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
