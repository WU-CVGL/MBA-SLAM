#ifndef SLAM_UTILS_CONFIGURATION_H
#define SLAM_UTILS_CONFIGURATION_H

#include <cereal/archives/json.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <filesystem>
#include <fstream>
#include <string>

namespace std
{
namespace filesystem
{
template<class Archive>
void CEREAL_LOAD_MINIMAL_FUNCTION_NAME(
    const Archive&, path& out, const std::string& in)
{
    out = in;
}

template<class Archive>
std::string CEREAL_SAVE_MINIMAL_FUNCTION_NAME(
    const Archive& ar, const path& p)
{
    return p;
}

} // namespace filesystem
} // namespace std

namespace SLAM {
namespace Utils {
struct Configuration
{
  public:
    std::string dataset_type = "unreal";
    std::filesystem::path root_path = "/home/trimbot/Desktop/peidong/Unreal4/ArchVizInterior_eth3d_camera_shake_3_trajectory";
    std::filesystem::path rel_path_to_im_list = "im_list.txt";
    std::filesystem::path rel_path_to_ref_ims = "rgb_sharp_middle";
    std::filesystem::path rel_path_to_depth_ims = "depth_sharp_middle";
    std::filesystem::path rel_path_to_curr_ims = "rgb_blur";
    std::filesystem::path rel_path_to_gt_poses = "groundtruth.txt";
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
        ar(CEREAL_NVP(keyframe_max_blur_kernel_mag));
        ar(CEREAL_NVP(with_gui));
        ar(CEREAL_NVP(frame_id_start_to_pause_imshow));
        ar(CEREAL_NVP(solver_type));
        ar(CEREAL_NVP(num_virtual_images_per_frame));
    }

    void make_template(const std::filesystem::path &filePath)
    {
        Configuration configs;
        std::ofstream ofs(filePath);
        cereal::JSONOutputArchive archive(ofs);
        archive(cereal::make_nvp("Configuration", configs));
    }

    static Configuration load_configs(const std::filesystem::path &filePath)
    {
        Configuration configs;
        std::ifstream ifs(filePath);
        cereal::JSONInputArchive archive(ifs);
        archive(cereal::make_nvp("Configuration", configs));
        return configs;
    }
};

} // namespace Utils
} // namespace SLAM

#endif // SLAM_UTILS_CONFIGURATION_H
