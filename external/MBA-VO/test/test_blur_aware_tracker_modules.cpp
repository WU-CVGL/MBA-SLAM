#include "BA_tracker/core/common/CustomType.h"
#include "BA_tracker/core/common/Random.h"
#include "BA_tracker/core/common/Spline.h"
#include "BA_tracker/core/common/Time.h"
#include "BA_tracker/core/image_proc/Gradient.h"
#include "BA_tracker/core/measurements/Image.h"
#include "BA_tracker/core/sensors/CameraPinhole.h"
#include "BA_tracker/core/states/Transformation.h"
#include "BA_tracker/utils/ImShow.h"
#include "BA_tracker/ba_tracker/compute_hessian_gradients_cost.h"
#include "BA_tracker/ba_tracker/compute_local_patches_xy.h"
#include "BA_tracker/ba_tracker/compute_pixel_intensity.h"
#include "BA_tracker/ba_tracker/compute_virtual_camera_poses.h"
#include "BA_tracker/ba_tracker/merge_hessian_gradient_cost.h"
#include "BA_tracker/ba_tracker/solve_normal_equation.h"
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace SLAM;
using namespace SLAM::Core;
using namespace SLAM::VO;

SplineSE3 *create_spline(double spline_t0, double spline_dt)
{
    Transformation T;
    Eigen::Quaterniond R;

    T.setRollPitchYaw(0.01 * M_PI, 0.01 * M_PI, 0.002 * M_PI);
    R = T.getRotation();
    Transformation T0(R, Eigen::Vector3d(0, 0, 0));

    T.setRollPitchYaw(0.02 * M_PI, 0.015 * M_PI, 0.0015 * M_PI);
    R = T.getRotation();
    Transformation T1(R, Eigen::Vector3d(5, 5, 0));

    T.setRollPitchYaw(0.03 * M_PI, 0.02 * M_PI, 0.001 * M_PI);
    R = T.getRotation();
    Transformation T2(R, Eigen::Vector3d(10, 10, 0));

    T.setRollPitchYaw(0.04 * M_PI, 0.025 * M_PI, 0.0005 * M_PI);
    R = T.getRotation();
    Transformation T3(R, Eigen::Vector3d(15, 15, 0));

    T.setRollPitchYaw(0.05 * M_PI, 0.03 * M_PI, 0.0 * M_PI);
    R = T.getRotation();
    Transformation T4(R, Eigen::Vector3d(20, 20, 0));

    T.setRollPitchYaw(0.05 * M_PI, 0.035 * M_PI, -0.0005 * M_PI);
    R = T.getRotation();
    Transformation T5(R, Eigen::Vector3d(25, 25, 0));

    T.setRollPitchYaw(0.07 * M_PI, 0.04 * M_PI, -0.001 * M_PI);
    R = T.getRotation();
    Transformation T6(R, Eigen::Vector3d(30, 30, 0));

    SplineSE3 *spline = new SplineSE3(spline_t0, spline_dt);
    spline->InsertControlKnot(T0.getRotation(), T0.getTranslation());
    spline->InsertControlKnot(T1.getRotation(), T1.getTranslation());
    spline->InsertControlKnot(T2.getRotation(), T2.getTranslation());
    spline->InsertControlKnot(T3.getRotation(), T3.getTranslation());
    spline->InsertControlKnot(T4.getRotation(), T4.getTranslation());
    spline->InsertControlKnot(T5.getRotation(), T5.getTranslation());
    spline->InsertControlKnot(T6.getRotation(), T6.getTranslation());

    return spline;
}

Image<unsigned char>::Ptr create_uniform_image(int H, int W)
{
    auto image = std::make_shared<Image<unsigned char>>(H, W, 1);

    for (int r = 0; r < H; r++)
    {
        for (int c = 0; c < W; c++)
        {
            image->getData(r, c)[0] = (c + r) % 255;
        }
    }
    return image;
}

void test_compute_pixel_intensity()
{
    std::cout << "\n-------------- " << __FUNCTION__ << " --------------\n";

    // create camera
    int H = 480;
    int W = 640;
    double fx = 320;
    double fy = 320;
    double cx = 320;
    double cy = 240;

    Eigen::Vector4d K(fx, fy, cx, cy);
    CameraPinhole camera(K, H, W);

    // find good ref_xy, pose and plane
    VectorX<double, 2> ref_xy;
    Vector2d cur_xy;
    double plane_depth;
    Eigen::Quaterniond R_c2r;
    Eigen::Vector3d t_c2r;

    bool is_found = false;

    while (!is_found)
    {
        ref_xy.values[0] = 20.5; //random_int(1, W - 1);
        ref_xy.values[1] = 20.5; //random_int(1, H - 1);

        plane_depth = random_float(5, 10);
        R_c2r = Eigen::Quaterniond::UnitRandom();
        t_c2r = Eigen::Vector3d::Random();

        Eigen::Vector3d P3dr;
        camera.unproject(Eigen::Vector2d(ref_xy.values[0], ref_xy.values[1]), plane_depth, P3dr);
        Eigen::Vector3d P3dc = R_c2r.inverse() * P3dr - R_c2r.inverse() * t_c2r;
        Eigen::Vector2d xy;
        is_found = camera.project(P3dc, xy);
        if (is_found)
        {
            cur_xy(0) = xy(0);
            cur_xy(1) = xy(1);
        }
    }

    // process test image
    Image<unsigned char>::Ptr I_ref = create_uniform_image(H, W);
    auto I_gradXY = std::make_shared<Image<float>>(H, W, 2);
    compute_image_gradients<unsigned char, float>(I_ref, I_gradXY, Image<float>::Ptr(nullptr));
    Vector3d I_dI(0, 0, 0);
    bilinear_interpolation<double>(I_ref->getData(), I_gradXY->getData(), H, W, ref_xy, I_dI);

    // compute analytical jacobian and interpolated pixel intensity
    double intensity;
    Eigen::Matrix<double, 1, 7> Ja, Jn;
    VectorX<double, 4> intrinsics;
    intrinsics.values[0] = fx;
    intrinsics.values[1] = fy;
    intrinsics.values[2] = cx;
    intrinsics.values[3] = cy;

    compute_pixel_intensity<double>(I_ref->getData(), I_gradXY->getData(), H, W, R_c2r.coeffs().data(), t_c2r.data(), plane_depth, intrinsics.values[0], intrinsics.values[1], intrinsics.values[2], intrinsics.values[3], cur_xy, &intensity, Ja.data());

    double max_intensity_err = fabs(I_dI(0) - intensity);
    std::cout << "max_intensity_err: " << max_intensity_err << "\n";
    if (max_intensity_err > 1e-4)
    {
        std::cout << "ground truth intensity: " << I_dI(0) << "\n";
        std::cout << "interpolated intensity: " << intensity << "\n\n";
    }

    // compute numerical jacobians
    double epsilon = 1e-6;
    for (int i = 0; i < 3; i++)
    {
        Eigen::Vector3d t_c2r_ = t_c2r;
        t_c2r_(i) += epsilon;

        double intensity_;
        compute_pixel_intensity<double>(I_ref->getData(), I_gradXY->getData(), H, W, R_c2r.coeffs().data(), t_c2r_.data(), plane_depth, intrinsics.values[0], intrinsics.values[1], intrinsics.values[2], intrinsics.values[3], cur_xy, &intensity_, nullptr);

        Jn(0, i) = (intensity_ - intensity) / epsilon;
    }

    for (int i = 0; i < 4; i++)
    {
        Eigen::Quaterniond R_c2r_ = R_c2r;
        R_c2r_.coeffs()(i) += epsilon;
        R_c2r_.normalize();

        double intensity_;
        compute_pixel_intensity<double>(I_ref->getData(), I_gradXY->getData(), H, W, R_c2r_.coeffs().data(), t_c2r.data(), plane_depth, intrinsics.values[0], intrinsics.values[1], intrinsics.values[2], intrinsics.values[3], cur_xy, &intensity_, nullptr);

        Jn(0, i + 3) = (intensity_ - intensity) / epsilon;
    }

    std::cout << "Analytical jacobians: " << Ja << "\n";
    std::cout << "Numerical jacobians : " << Jn << "\n";
}

void test_compute_virtual_camera_poses()
{
    std::cout << "\n-------------- " << __FUNCTION__ << " --------------\n";

    typedef double T;

    double spline_start_time = 0;
    double spline_sample_interval = 0.5;
    SplineSE3 *spline = create_spline(spline_start_time, spline_sample_interval);

    T *cuda_spline_data_t = nullptr;
    T *cuda_spline_data_R = nullptr;

    cudaMalloc((void **)&cuda_spline_data_t, sizeof(T) * spline->getNumCtrlKnots() * 3);
    cudaMalloc((void **)&cuda_spline_data_R, sizeof(T) * spline->getNumCtrlKnots() * 4);
    cudaMemcpy(cuda_spline_data_t,
               spline->get_knot_data_t(),
               sizeof(T) * spline->getNumCtrlKnots() * 3,
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_spline_data_R,
               spline->get_knot_data_R(),
               sizeof(T) * spline->getNumCtrlKnots() * 4,
               cudaMemcpyHostToDevice);

    // create capture time and exposure time
    double frame_t0 = 0.25;
    double frame_sample_t = 0.5;
    double frame_exp_t = 0.1;
    std::vector<double> img_cap_time;
    std::vector<double> img_exp_time;
    img_cap_time.push_back(frame_t0);
    img_cap_time.push_back(frame_t0 + frame_sample_t);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 2);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 3);

    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);

    T *cuda_img_cap_time = nullptr;
    T *cuda_img_exp_time = nullptr;

    cudaMalloc((void **)&cuda_img_cap_time, sizeof(T) * img_cap_time.size());
    cudaMalloc((void **)&cuda_img_exp_time, sizeof(T) * img_exp_time.size());
    cudaMemcpy(cuda_img_cap_time, img_cap_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_img_exp_time, img_exp_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);

    // create temporaray storage for generated data etc.
#define N 32
    int N_virtual_poses = N * img_cap_time.size();
    T *cuda_sampled_virtual_poses = nullptr;
    T *cuda_jacobian_virtual_pose_t_to_ctrl_knots = nullptr;
    T *cuda_jacobian_virtual_pose_R_to_ctrl_knots = nullptr;
    T *jacobian_log_exp = nullptr;
    T *temp_X_4x4 = nullptr;
    T *temp_Y_4x4 = nullptr;
    T *temp_Z_4x4 = nullptr;

    cudaMalloc((void **)&cuda_sampled_virtual_poses, sizeof(T) * N_virtual_poses * 7);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_t_to_ctrl_knots, sizeof(T) * N_virtual_poses * 36);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_R_to_ctrl_knots, sizeof(T) * N_virtual_poses * 48);
    cudaMalloc((void **)&jacobian_log_exp, sizeof(T) * N_virtual_poses * 72);
    cudaMalloc((void **)&temp_X_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Y_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Z_4x4, sizeof(T) * N_virtual_poses * 16);

    compute_virtual_camera_poses(N,
                                 img_cap_time.size(),
                                 cuda_img_cap_time,
                                 cuda_img_exp_time,
                                 4,
                                 spline_start_time,
                                 spline_sample_interval,
                                 cuda_spline_data_t,
                                 cuda_spline_data_R,
                                 cuda_sampled_virtual_poses,
                                 cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                 cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                 jacobian_log_exp,
                                 temp_X_4x4,
                                 temp_Y_4x4,
                                 temp_Z_4x4);

    // transfer to cpu
    T *cpu_sampled_virtual_poses = new T[N_virtual_poses * 7];
    T *cpu_jacobian_virtual_pose_t_to_ctrl_knots = new T[N_virtual_poses * 36];
    T *cpu_jacobian_virtual_pose_R_to_ctrl_knots = new T[N_virtual_poses * 48];

    cudaMemcpy(cpu_sampled_virtual_poses,
               cuda_sampled_virtual_poses,
               sizeof(double) * N_virtual_poses * 7,
               cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu_jacobian_virtual_pose_t_to_ctrl_knots,
               cuda_jacobian_virtual_pose_t_to_ctrl_knots,
               sizeof(double) * N_virtual_poses * 36,
               cudaMemcpyDeviceToHost);

    cudaMemcpy(cpu_jacobian_virtual_pose_R_to_ctrl_knots,
               cuda_jacobian_virtual_pose_R_to_ctrl_knots,
               sizeof(double) * N_virtual_poses * 48,
               cudaMemcpyDeviceToHost);

    // verify pose
    int global_pose_idx = random_int(0, N_virtual_poses - 1);
    int frame_idx = global_pose_idx / N;
    int local_pose_idx = global_pose_idx % N;
    double t = frame_t0 + frame_idx * frame_sample_t - frame_exp_t * 0.5 + local_pose_idx * frame_exp_t / (N - 1);

    Eigen::Quaterniond R_cpu;
    Eigen::Vector3d t_cpu;
    Eigen::Matrix<double, 3, 12, Eigen::RowMajor> jacobian_t_cpu;
    Eigen::Matrix<double, 4, 12, Eigen::RowMajor> jacobian_R_cpu;

    spline->getPose(t, R_cpu, t_cpu, jacobian_R_cpu.data(), jacobian_t_cpu.data());

    Eigen::Vector3d t_gpu(cpu_sampled_virtual_poses + global_pose_idx * 7);
    Eigen::Vector4d R_gpu(cpu_sampled_virtual_poses + global_pose_idx * 7 + 3);

    double max_pose_t_error = (t_gpu - t_cpu).cwiseAbs().maxCoeff();
    double max_pose_R_error = (R_gpu - R_cpu.coeffs()).cwiseAbs().maxCoeff();

    std::cout << "max pose err: " << max_pose_t_error << " " << max_pose_R_error << "\n";
    if (max_pose_t_error > 1e-4 || max_pose_R_error > 1e-4)
    {
        std::cout << "cpu_pose_t: " << t_cpu.transpose() << "\n";
        std::cout << "gpu_pose_t: " << t_gpu.transpose() << "\n";

        std::cout << "cpu_pose_R: " << R_cpu.coeffs().transpose() << "\n";
        std::cout << "cpu_pose_R: " << R_gpu.transpose() << "\n";
    }

    // verify jacobian
    Eigen::Map<Eigen::Matrix<double, 3, 12, Eigen::RowMajor>> jacobian_t_gpu(cpu_jacobian_virtual_pose_t_to_ctrl_knots + global_pose_idx * 36);
    Eigen::Matrix<double, 4, 12, Eigen::RowMajor> jacobian_R_gpu(cpu_jacobian_virtual_pose_R_to_ctrl_knots + global_pose_idx * 48);

    double max_jacob_t_err = (jacobian_t_cpu - jacobian_t_gpu).cwiseAbs().maxCoeff();
    double max_jacob_R_err = (jacobian_R_cpu - jacobian_R_gpu).cwiseAbs().maxCoeff();
    std::cout << "max_jacob_err: " << max_jacob_t_err << " " << max_jacob_R_err << "\n";

    if (max_jacob_t_err > 1e-4)
    {
        std::cout << "CPU Jacob_t: \n";
        std::cout << jacobian_t_cpu << "\n";

        std::cout << "GPU Jacob_t: \n";
        std::cout << jacobian_t_gpu << "\n\n";
    }

    if (max_jacob_R_err > 1e-4)
    {
        std::cout << "CPU Jacob_R: \n";
        std::cout << jacobian_R_cpu << "\n";

        std::cout << "GPU Jacob_R: \n";
        std::cout << jacobian_R_gpu << "\n";
    }
#undef N
}

void test_compute_local_patches()
{
    std::cout << "\n-------------- " << __FUNCTION__ << " --------------\n";

    typedef double T;

    double spline_start_time = 0;
    double spline_sample_interval = 0.5;
    SplineSE3 *spline = create_spline(spline_start_time, spline_sample_interval);

    T *cuda_spline_data_t = nullptr;
    T *cuda_spline_data_R = nullptr;

    cudaMalloc((void **)&cuda_spline_data_t, sizeof(T) * spline->getNumCtrlKnots() * 3);
    cudaMalloc((void **)&cuda_spline_data_R, sizeof(T) * spline->getNumCtrlKnots() * 4);
    cudaMemcpy(cuda_spline_data_t,
               spline->get_knot_data_t(),
               sizeof(T) * spline->getNumCtrlKnots() * 3,
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_spline_data_R,
               spline->get_knot_data_R(),
               sizeof(T) * spline->getNumCtrlKnots() * 4,
               cudaMemcpyHostToDevice);

    // create capture time and exposure time
    double frame_t0 = 0.25;
    double frame_sample_t = 0.5;
    double frame_exp_t = 0.1;
    std::vector<double> img_cap_time;
    std::vector<double> img_exp_time;
    img_cap_time.push_back(frame_t0);
    img_cap_time.push_back(frame_t0 + frame_sample_t);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 2);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 3);

    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);

    T *cuda_img_cap_time = nullptr;
    T *cuda_img_exp_time = nullptr;

    cudaMalloc((void **)&cuda_img_cap_time, sizeof(T) * img_cap_time.size());
    cudaMalloc((void **)&cuda_img_exp_time, sizeof(T) * img_exp_time.size());
    cudaMemcpy(cuda_img_cap_time, img_cap_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_img_exp_time, img_exp_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);

    // create temporaray storage for generated data etc.
#define N 32
    int N_virtual_poses = N * img_cap_time.size();
    T *cuda_sampled_virtual_poses = nullptr;
    T *cuda_jacobian_virtual_pose_t_to_ctrl_knots = nullptr;
    T *cuda_jacobian_virtual_pose_R_to_ctrl_knots = nullptr;
    T *jacobian_log_exp = nullptr;
    T *temp_X_4x4 = nullptr;
    T *temp_Y_4x4 = nullptr;
    T *temp_Z_4x4 = nullptr;

    cudaMalloc((void **)&cuda_sampled_virtual_poses, sizeof(T) * N_virtual_poses * 7);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_t_to_ctrl_knots, sizeof(T) * N_virtual_poses * 36);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_R_to_ctrl_knots, sizeof(T) * N_virtual_poses * 48);
    cudaMalloc((void **)&jacobian_log_exp, sizeof(T) * N_virtual_poses * 72);
    cudaMalloc((void **)&temp_X_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Y_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Z_4x4, sizeof(T) * N_virtual_poses * 16);

    compute_virtual_camera_poses(N,
                                 img_cap_time.size(),
                                 cuda_img_cap_time,
                                 cuda_img_exp_time,
                                 4,
                                 spline_start_time,
                                 spline_sample_interval,
                                 cuda_spline_data_t,
                                 cuda_spline_data_R,
                                 cuda_sampled_virtual_poses,
                                 cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                 cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                 jacobian_log_exp,
                                 temp_X_4x4,
                                 temp_Y_4x4,
                                 temp_Z_4x4);

    //
    VectorX<int, 2> im_HW;
    VectorX<double, 4> intrinsics;
    im_HW.values[0] = 480;
    im_HW.values[1] = 640;
    intrinsics.values[0] = 320;
    intrinsics.values[1] = 320;
    intrinsics.values[2] = 320;
    intrinsics.values[3] = 240;

    std::vector<Vector2d> sparse_keypoints;
    std::vector<double> sparse_keypoints_z;
    for (int i = 0; i < 145; i++)
    {
        sparse_keypoints.push_back(Vector2d(random_int(20, 620), random_int(20, 460)));
        sparse_keypoints_z.push_back(random_float(20, 45));
    }

    Vector2d *cuda_sparse_keypoints;
    double *cuda_sparse_keypoints_z;
    cudaMalloc((void **)&cuda_sparse_keypoints, sizeof(Vector2d) * sparse_keypoints.size());
    cudaMalloc((void **)&cuda_sparse_keypoints_z, sizeof(double) * sparse_keypoints_z.size());
    cudaMemcpy(cuda_sparse_keypoints, sparse_keypoints.data(), sizeof(Vector2d) * sparse_keypoints.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_sparse_keypoints_z, sparse_keypoints_z.data(), sizeof(double) * sparse_keypoints_z.size(), cudaMemcpyHostToDevice);

    Vector2d *cuda_local_patches;
    cudaMalloc((void **)&cuda_local_patches, sizeof(Vector2d) * img_cap_time.size() * sparse_keypoints.size());

    compute_local_patches_xy(N,
                             img_cap_time.size(),
                             cuda_sampled_virtual_poses,
                             cuda_sparse_keypoints,
                             cuda_sparse_keypoints_z,
                             sparse_keypoints.size(),
                             intrinsics,
                             im_HW,
                             cuda_local_patches);

    Vector2d *cpu_local_patches = new Vector2d[img_cap_time.size() * sparse_keypoints.size()];
    cudaMemcpy(cpu_local_patches, cuda_local_patches, sizeof(Vector2d) * img_cap_time.size() * sparse_keypoints.size(), cudaMemcpyDeviceToHost);

    // verify local_patch_xy
    const int global_patch_idx = random_int(0, img_cap_time.size() * sparse_keypoints.size());
    const int frame_idx = global_patch_idx / sparse_keypoints.size();
    const int kpt_idx = global_patch_idx % sparse_keypoints.size();

    const double frame_t = frame_t0 + frame_idx * frame_sample_t + frame_exp_t / (N - 1) * 0.5;
    Eigen::Quaterniond R_c2r;
    Eigen::Vector3d t_c2r;
    spline->getPose(frame_t, R_c2r, t_c2r);

    Vector2d kpt_xy = sparse_keypoints[kpt_idx];
    double kpt_z = sparse_keypoints_z[kpt_idx];

    Eigen::Vector3d P3dr;
    P3dr(0) = (kpt_xy(0) - intrinsics.values[2]) / intrinsics.values[0] * kpt_z;
    P3dr(1) = (kpt_xy(1) - intrinsics.values[3]) / intrinsics.values[1] * kpt_z;
    P3dr(2) = kpt_z;

    Eigen::Vector3d P3dc = R_c2r.inverse() * P3dr - R_c2r.inverse() * t_c2r;
    Eigen::Vector2d p2dc;
    p2dc(0) = P3dc(0) / P3dc(2);
    p2dc(1) = P3dc(1) / P3dc(2);
    p2dc(0) = intrinsics.values[0] * p2dc(0) + intrinsics.values[2];
    p2dc(1) = intrinsics.values[1] * p2dc(1) + intrinsics.values[3];

    std::cout << "CPU xy: " << p2dc.transpose() << "\n";
    std::cout << "GPU xy: "
              << cpu_local_patches[global_patch_idx](0) << " "
              << cpu_local_patches[global_patch_idx](1) << "\n\n";

#undef N
}

void test_compute_pixel_jacobian_residual()
{
    std::cout << "\n-------------- " << __FUNCTION__ << " --------------\n";

    typedef double T;

    double spline_start_time = 0;
    double spline_sample_interval = 0.5;
    SplineSE3 *spline = create_spline(spline_start_time, spline_sample_interval);

    T *cuda_spline_data_t = nullptr;
    T *cuda_spline_data_R = nullptr;

    cudaMalloc((void **)&cuda_spline_data_t, sizeof(T) * spline->getNumCtrlKnots() * 3);
    cudaMalloc((void **)&cuda_spline_data_R, sizeof(T) * spline->getNumCtrlKnots() * 4);
    cudaMemcpy(cuda_spline_data_t,
               spline->get_knot_data_t(),
               sizeof(T) * spline->getNumCtrlKnots() * 3,
               cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_spline_data_R,
               spline->get_knot_data_R(),
               sizeof(T) * spline->getNumCtrlKnots() * 4,
               cudaMemcpyHostToDevice);

    // create capture time and exposure time
    double frame_t0 = 0.25;
    double frame_sample_t = 0.5;
    double frame_exp_t = 0.1;
    std::vector<double> img_cap_time;
    std::vector<double> img_exp_time;
    img_cap_time.push_back(frame_t0);
    img_cap_time.push_back(frame_t0 + frame_sample_t);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 2);
    img_cap_time.push_back(frame_t0 + frame_sample_t * 3);

    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);
    img_exp_time.push_back(frame_exp_t);

    T *cuda_img_cap_time = nullptr;
    T *cuda_img_exp_time = nullptr;

    cudaMalloc((void **)&cuda_img_cap_time, sizeof(T) * img_cap_time.size());
    cudaMalloc((void **)&cuda_img_exp_time, sizeof(T) * img_exp_time.size());
    cudaMemcpy(cuda_img_cap_time, img_cap_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_img_exp_time, img_exp_time.data(), sizeof(double) * img_cap_time.size(), cudaMemcpyHostToDevice);

    // create temporaray storage for generated data etc.
#define N 32
    int N_virtual_poses = N * img_cap_time.size();
    T *cuda_sampled_virtual_poses = nullptr;
    T *cuda_jacobian_virtual_pose_t_to_ctrl_knots = nullptr;
    T *cuda_jacobian_virtual_pose_R_to_ctrl_knots = nullptr;
    T *jacobian_log_exp = nullptr;
    T *temp_X_4x4 = nullptr;
    T *temp_Y_4x4 = nullptr;
    T *temp_Z_4x4 = nullptr;

    cudaMalloc((void **)&cuda_sampled_virtual_poses, sizeof(T) * N_virtual_poses * 7);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_t_to_ctrl_knots, sizeof(T) * N_virtual_poses * 36);
    cudaMalloc((void **)&cuda_jacobian_virtual_pose_R_to_ctrl_knots, sizeof(T) * N_virtual_poses * 48);
    cudaMalloc((void **)&jacobian_log_exp, sizeof(T) * N_virtual_poses * 72);
    cudaMalloc((void **)&temp_X_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Y_4x4, sizeof(T) * N_virtual_poses * 16);
    cudaMalloc((void **)&temp_Z_4x4, sizeof(T) * N_virtual_poses * 16);

    compute_virtual_camera_poses(N,
                                 img_cap_time.size(),
                                 cuda_img_cap_time,
                                 cuda_img_exp_time,
                                 4,
                                 spline_start_time,
                                 spline_sample_interval,
                                 cuda_spline_data_t,
                                 cuda_spline_data_R,
                                 cuda_sampled_virtual_poses,
                                 cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                 cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                 jacobian_log_exp,
                                 temp_X_4x4,
                                 temp_Y_4x4,
                                 temp_Z_4x4);

    double *cpu_sampled_virtual_poses = new double[N_virtual_poses * 7];
    cudaMemcpy(cpu_sampled_virtual_poses, cuda_sampled_virtual_poses, sizeof(double) * N_virtual_poses * 7, cudaMemcpyDeviceToHost);

    //
    const int H = 480;
    const int W = 640;
    const double fx = 320;
    const double fy = 320;
    const double cx = 320;
    const double cy = 240;

    VectorX<int, 2> im_HW;
    VectorX<double, 4> intrinsics;
    im_HW.values[0] = H;
    im_HW.values[1] = W;
    intrinsics.values[0] = fx;
    intrinsics.values[1] = fy;
    intrinsics.values[2] = cx;
    intrinsics.values[3] = cy;

    std::vector<Vector2d> sparse_keypoints;
    std::vector<double> sparse_keypoints_z;
    for (int i = 0; i < 145; i++)
    {
        sparse_keypoints.push_back(Vector2d(random_int(20, 620) + 0.1, random_int(20, 460) + 0.1));
        sparse_keypoints_z.push_back(random_float(20, 45));
    }

    Vector2d *cuda_sparse_keypoints;
    double *cuda_sparse_keypoints_z;
    cudaMalloc((void **)&cuda_sparse_keypoints, sizeof(Vector2d) * sparse_keypoints.size());
    cudaMalloc((void **)&cuda_sparse_keypoints_z, sizeof(double) * sparse_keypoints_z.size());
    cudaMemcpy(cuda_sparse_keypoints, sparse_keypoints.data(), sizeof(Vector2d) * sparse_keypoints.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_sparse_keypoints_z, sparse_keypoints_z.data(), sizeof(double) * sparse_keypoints_z.size(), cudaMemcpyHostToDevice);

    Vector2d *cuda_local_patches;
    cudaMalloc((void **)&cuda_local_patches, sizeof(Vector2d) * img_cap_time.size() * sparse_keypoints.size());

    compute_local_patches_xy(N,
                             img_cap_time.size(),
                             cuda_sampled_virtual_poses,
                             cuda_sparse_keypoints,
                             cuda_sparse_keypoints_z,
                             sparse_keypoints.size(),
                             intrinsics,
                             im_HW,
                             cuda_local_patches);

    Vector2d *cpu_local_patches = new Vector2d[img_cap_time.size() * sparse_keypoints.size()];
    cudaMemcpy(cpu_local_patches, cuda_local_patches, sizeof(Vector2d) * img_cap_time.size() * sparse_keypoints.size(), cudaMemcpyDeviceToHost);

    // read in reference images
    Image<unsigned char>::Ptr I_ref = create_uniform_image(H, W);
    auto I_gradXY = std::make_shared<Image<float>>(H, W, 2);
    compute_image_gradients<unsigned char, float>(I_ref, I_gradXY, Image<float>::Ptr(nullptr));

    I_ref->uploadToGpu();
    I_gradXY->uploadToGpu();

    // get current images
    const int num_frames = img_cap_time.size();
    std::vector<unsigned char *> gpu_I_cur_imgs;
    std::vector<unsigned char *> cpu_I_cur_imgs;
    for (int i = 0; i < num_frames; i++)
    {
        Image<unsigned char> *I_cur = new Image<unsigned char>(H, W, 1);
        I_cur->copyFrom(I_ref->getData(), H, W, 1);
        I_cur->uploadToGpu();
        gpu_I_cur_imgs.push_back(I_cur->getGpuData());
        cpu_I_cur_imgs.push_back(I_cur->getData());
    }
    unsigned char **cuda_I_cur_imgs;
    cudaMalloc((void **)&cuda_I_cur_imgs, sizeof(void *) * num_frames);
    cudaMemcpy(cuda_I_cur_imgs, gpu_I_cur_imgs.data(), sizeof(void *) * num_frames, cudaMemcpyHostToDevice);

    // compute per pixel residual and jacobians
    const int patch_size = 8;
    int *local_patch_pattern_xy = new int[patch_size * 2];
    local_patch_pattern_xy[0] = -2;
    local_patch_pattern_xy[1] = -2;
    local_patch_pattern_xy[2] = 2;
    local_patch_pattern_xy[3] = -2;
    local_patch_pattern_xy[4] = -1;
    local_patch_pattern_xy[5] = -1;
    local_patch_pattern_xy[6] = 1;
    local_patch_pattern_xy[7] = -1;
    local_patch_pattern_xy[8] = 0;
    local_patch_pattern_xy[9] = 0;
    local_patch_pattern_xy[10] = 0;
    local_patch_pattern_xy[11] = 1;
    local_patch_pattern_xy[12] = -2;
    local_patch_pattern_xy[13] = 2;
    local_patch_pattern_xy[14] = 2;
    local_patch_pattern_xy[15] = 2;
    int *cuda_local_patch_pattern_xy;
    cudaMalloc((void **)&cuda_local_patch_pattern_xy, sizeof(int) * 2 * patch_size);
    cudaMemcpy(cuda_local_patch_pattern_xy, local_patch_pattern_xy, sizeof(int) * 2 * patch_size, cudaMemcpyHostToDevice);

    double *cuda_pixel_residuals;
    double *cuda_pixel_jacobians_tR;
    FLOAT *cuda_vir_pixel_jacobians_tR;
    const int num_keypoints = sparse_keypoints.size();
    cudaMalloc((void **)&cuda_pixel_residuals, sizeof(double) * num_frames * num_keypoints * patch_size);
    cudaMalloc((void **)&cuda_pixel_jacobians_tR, sizeof(double) * num_frames * num_keypoints * patch_size * 24);
    cudaMalloc((void **)&cuda_vir_pixel_jacobians_tR, sizeof(FLOAT) * num_frames * num_keypoints * patch_size * 24 * N);

    compute_pixel_jacobian_residual(I_ref->getGpuData(),
                                    I_gradXY->getGpuData(),
                                    cuda_I_cur_imgs,
                                    N,
                                    num_frames,
                                    cuda_sampled_virtual_poses,
                                    4,
                                    cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                    cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                    cuda_local_patches,
                                    cuda_sparse_keypoints_z,
                                    num_keypoints,
                                    cuda_local_patch_pattern_xy,
                                    patch_size,
                                    intrinsics,
                                    im_HW,
                                    cuda_vir_pixel_jacobians_tR,
                                    cuda_pixel_residuals,
                                    cuda_pixel_jacobians_tR);

    const int num_pixels = num_frames * num_keypoints * patch_size;
    double *cpu_pixel_residuals = new double[num_pixels];
    double *cpu_pixel_jacobians_tR = new double[num_pixels * 24];
    cudaMemcpy(cpu_pixel_residuals, cuda_pixel_residuals, sizeof(double) * num_pixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_pixel_jacobians_tR, cuda_pixel_jacobians_tR, sizeof(double) * num_pixels * 24, cudaMemcpyDeviceToHost);

    // TEST
    const int frame_idx = 2;   //random_int(0, num_frames);
    const int patch_idx = 100; //random_int(0, num_keypoints);
    const int pixel_idx = 1;   //random_int(0, num_pixels);
    const int global_pixel_idx = frame_idx * num_keypoints * patch_size + patch_idx * patch_size + pixel_idx;

    Vector2d cur_xy = cpu_local_patches[frame_idx * num_keypoints + patch_idx];
    int dx = local_patch_pattern_xy[pixel_idx * 2];
    int dy = local_patch_pattern_xy[pixel_idx * 2 + 1];
    cur_xy(0) += dx;
    cur_xy(1) += dy;
    cur_xy(0) = int(cur_xy(0));
    cur_xy(1) = int(cur_xy(1));
    double plane_depth = sparse_keypoints_z[patch_idx];
    double intensity = 0;
    for (int i = 0; i < N; i++)
    {
        double *T_c2r = cpu_sampled_virtual_poses + frame_idx * N * 7 + i * 7;
        double intensity_;
        compute_pixel_intensity<double>(I_ref->getData(),
                                        nullptr,
                                        im_HW.values[0],
                                        im_HW.values[1],
                                        T_c2r + 3,
                                        T_c2r,
                                        plane_depth,
                                        intrinsics.values[0],
                                        intrinsics.values[1],
                                        intrinsics.values[2],
                                        intrinsics.values[3],
                                        cur_xy,
                                        &intensity_,
                                        nullptr);
        intensity += intensity_ / float(N);
    }

    // printf("cur_xy %f %f\n", cur_xy(0), cur_xy(1));

    Vector3d I_and_dI;
    unsigned char *I_cur = cpu_I_cur_imgs[frame_idx];
    bilinear_interpolation<double>(I_cur, nullptr, im_HW.values[0], im_HW.values[1], cur_xy, I_and_dI);

    double cpu_residual = intensity - I_and_dI(0);

    std::cout << "cpu_residual: " << cpu_residual << "\n";
    std::cout << "gpu_residual: " << cpu_pixel_residuals[global_pixel_idx] << "\n\n";

    // TEST jacobians
    std::cout << "analytical_jacobian_tR: " << Eigen::Map<Eigen::Matrix<double, 1, 24>>(cpu_pixel_jacobians_tR + global_pixel_idx * 24) << "\n";

    // compute numerical jacobians
    Eigen::Matrix<double, 24, 1> Jn_tR;

    double u;
    const double epsilon = 1e-4;
    int ctrl_knot_start_idx = 0;
    SplineSegmentStartKnotIdxAndNormalizedU(frame_idx * frame_sample_t + frame_t0, spline_start_time, spline_sample_interval, ctrl_knot_start_idx, u);
    double dt[12];
    double dR[12];
    for (int i = 0; i < 12; i++)
    {
        std::fill(dt, dt + 12, 0);
        dt[i] += epsilon;
        SplineSE3::Ptr spline_ = spline->clone();
        spline_->UpdateCtrlKnot_t(ctrl_knot_start_idx, 4, dt);

        cudaMemcpy(cuda_spline_data_t,
                   spline_->get_knot_data_t(),
                   sizeof(T) * spline_->getNumCtrlKnots() * 3,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_spline_data_R,
                   spline_->get_knot_data_R(),
                   sizeof(T) * spline_->getNumCtrlKnots() * 4,
                   cudaMemcpyHostToDevice);

        compute_virtual_camera_poses(N,
                                     img_cap_time.size(),
                                     cuda_img_cap_time,
                                     cuda_img_exp_time,
                                     4,
                                     spline_start_time,
                                     spline_sample_interval,
                                     cuda_spline_data_t,
                                     cuda_spline_data_R,
                                     cuda_sampled_virtual_poses,
                                     cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                     cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                     jacobian_log_exp,
                                     temp_X_4x4,
                                     temp_Y_4x4,
                                     temp_Z_4x4);

        compute_pixel_jacobian_residual(I_ref->getGpuData(),
                                        I_gradXY->getGpuData(),
                                        cuda_I_cur_imgs,
                                        N,
                                        num_frames,
                                        cuda_sampled_virtual_poses,
                                        4,
                                        cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                        cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                        cuda_local_patches,
                                        cuda_sparse_keypoints_z,
                                        num_keypoints,
                                        cuda_local_patch_pattern_xy,
                                        patch_size,
                                        intrinsics,
                                        im_HW,
                                        cuda_vir_pixel_jacobians_tR,
                                        cuda_pixel_residuals,
                                        cuda_pixel_jacobians_tR);

        cudaMemcpy(cpu_pixel_residuals, cuda_pixel_residuals, sizeof(double) * num_pixels, cudaMemcpyDeviceToHost);
        double cpu_residual_ = cpu_pixel_residuals[global_pixel_idx];
        Jn_tR(i, 0) = (cpu_residual_ - cpu_residual) / epsilon;
    }

    for (int i = 0; i < 12; i++)
    {
        std::fill(dR, dR + 12, 0);
        dR[i] += epsilon;
        SplineSE3::Ptr spline_ = spline->clone();
        spline_->UpdateCtrlKnot_R(ctrl_knot_start_idx, 4, dR);

        cudaMemcpy(cuda_spline_data_t,
                   spline_->get_knot_data_t(),
                   sizeof(T) * spline_->getNumCtrlKnots() * 3,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_spline_data_R,
                   spline_->get_knot_data_R(),
                   sizeof(T) * spline_->getNumCtrlKnots() * 4,
                   cudaMemcpyHostToDevice);

        compute_virtual_camera_poses(N,
                                     img_cap_time.size(),
                                     cuda_img_cap_time,
                                     cuda_img_exp_time,
                                     4,
                                     spline_start_time,
                                     spline_sample_interval,
                                     cuda_spline_data_t,
                                     cuda_spline_data_R,
                                     cuda_sampled_virtual_poses,
                                     cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                     cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                     jacobian_log_exp,
                                     temp_X_4x4,
                                     temp_Y_4x4,
                                     temp_Z_4x4);

        compute_pixel_jacobian_residual(I_ref->getGpuData(),
                                        I_gradXY->getGpuData(),
                                        cuda_I_cur_imgs,
                                        N,
                                        num_frames,
                                        cuda_sampled_virtual_poses,
                                        4,
                                        cuda_jacobian_virtual_pose_t_to_ctrl_knots,
                                        cuda_jacobian_virtual_pose_R_to_ctrl_knots,
                                        cuda_local_patches,
                                        cuda_sparse_keypoints_z,
                                        num_keypoints,
                                        cuda_local_patch_pattern_xy,
                                        patch_size,
                                        intrinsics,
                                        im_HW,
                                        cuda_vir_pixel_jacobians_tR,
                                        cuda_pixel_residuals,
                                        cuda_pixel_jacobians_tR);

        cudaMemcpy(cpu_pixel_residuals, cuda_pixel_residuals, sizeof(double) * num_pixels, cudaMemcpyDeviceToHost);
        double cpu_residual_ = cpu_pixel_residuals[global_pixel_idx];
        Jn_tR(i + 12, 0) = (cpu_residual_ - cpu_residual) / epsilon;
    }
    std::cout << "numerical_jacobian_tR : " << Jn_tR.transpose() << "\n";

#undef N
}

void test_compute_patch_cost_gradient_hessian()
{
    std::cout << "\n---------------" << __FUNCTION__ << "----------------\n";
    const int num_frames = 5;
    const int num_keypoints = 145;
    const int patch_size = 8;
    const int num_pixels = num_frames * num_keypoints * patch_size;
    const int num_patches = num_frames * num_keypoints;

    double *cpu_pixel_residuals = new double[num_pixels];
    double *cpu_pixel_jacobians = new double[num_pixels * 24];
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>
        eigen_cpu_pixel_residuals(cpu_pixel_residuals, num_pixels);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>
        eigen_cpu_pixel_jacobians(cpu_pixel_jacobians, num_pixels * 24);
    eigen_cpu_pixel_residuals.setRandom();
    eigen_cpu_pixel_jacobians.setRandom();

    double *gpu_pixel_residuals;
    double *gpu_pixel_jacobians;
    cudaMalloc((void **)&gpu_pixel_residuals, sizeof(double) * num_pixels);
    cudaMalloc((void **)&gpu_pixel_jacobians, sizeof(double) * num_pixels * 24);
    cudaMemcpy(gpu_pixel_residuals, cpu_pixel_residuals, sizeof(double) * num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pixel_jacobians, cpu_pixel_jacobians, sizeof(double) * num_pixels * 24, cudaMemcpyHostToDevice);

    // compute cost, gradient, hessian with GPU
    const double huber_k = 0.1;
    double *gpu_patch_cost_gradient_hessian;
    cudaMalloc((void **)&gpu_patch_cost_gradient_hessian, sizeof(double) * num_patches * 325);

    compute_patch_cost_gradient_hessian(num_frames,
                                        num_keypoints,
                                        patch_size,
                                        4,
                                        gpu_pixel_residuals,
                                        gpu_pixel_jacobians,
                                        huber_k,
                                        1,
                                        gpu_patch_cost_gradient_hessian);

    double *cpu_patch_cost_gradient_hessian = new double[num_patches * 325];
    cudaMemcpy(cpu_patch_cost_gradient_hessian, gpu_patch_cost_gradient_hessian, sizeof(double) * num_patches * 325, cudaMemcpyDeviceToHost);

    // compute gpu results
    const int patch_idx = 96;
    const int offset = patch_idx * 325;
    double gpu_cost = cpu_patch_cost_gradient_hessian[offset];
    Eigen::Matrix<double, 24, 1> gpu_gradient;
    Eigen::Matrix<double, 24, 24> gpu_hessian;
    int shift = 25;
    for (int i = 0; i < 24; i++)
    {
        gpu_gradient(i) = cpu_patch_cost_gradient_hessian[offset + i + 1];
        for (int j = i; j < 24; j++)
        {
            double value = cpu_patch_cost_gradient_hessian[offset + shift++];
            gpu_hessian(i, j) = value;
            gpu_hessian(j, i) = value;
        }
    }

    // compute cpu results
    double cpu_cost = 0;
    Eigen::Matrix<double, 24, 1> cpu_gradient;
    Eigen::Matrix<double, 24, 24> cpu_hessian;
    cpu_gradient.setZero();
    cpu_hessian.setZero();
    for (int i = 0; i < patch_size; i++)
    {
        const int pixel_idx = patch_idx * patch_size + i;
        double r = cpu_pixel_residuals[pixel_idx];
        Eigen::Matrix<double, 24, 1> J(cpu_pixel_jacobians + pixel_idx * 24);

        const double half_rr = 0.5 * r * r;
        double rho = half_rr;
        double drho_dx = 1;
        if (half_rr > huber_k * huber_k)
        {
            rho = 2 * huber_k * sqrtf(half_rr) - huber_k * huber_k;
            drho_dx = huber_k / sqrtf(half_rr);
        }

        cpu_cost += rho;
        cpu_gradient += drho_dx * r * J;
        cpu_hessian += drho_dx * J * J.transpose();
    }

    double max_cost_err = fabs(cpu_cost - gpu_cost);
    double max_grad_err = (cpu_gradient - gpu_gradient).cwiseAbs().maxCoeff();
    double max_hess_err = (cpu_hessian - gpu_hessian).cwiseAbs().maxCoeff();

    std::cout << "max_cost_err: " << max_cost_err << "\n";
    if (max_cost_err > 1e-8)
    {
        std::cout << "cpu_cost: " << cpu_cost << "\n";
        std::cout << "gpu_cost: " << gpu_cost << "\n\n";
    }

    std::cout << "max_grad_err: " << max_grad_err << "\n";
    if (max_grad_err > 1e-6)
    {
        std::cout << "cpu_gradient: " << cpu_gradient.transpose() << "\n";
        std::cout << "gpu_gradient: " << gpu_gradient.transpose() << "\n\n";
    }

    std::cout << "max_hess_err: " << max_hess_err << "\n";
    if (max_hess_err > 1e-6)
    {
        std::cout << "cpu_hessian: \n";
        std::cout << cpu_hessian << "\n";

        std::cout << "gpu_hessian: \n";
        std::cout << gpu_hessian << "\n";
    }
}

void test_compute_frame_cost_gradient_hessian()
{
    std::cout << "\n---------------" << __FUNCTION__ << "----------------\n";
    const int num_frames = 6;
    const int num_keypoints = 145;
    const int size = 325;
    const int num_elems = num_frames * num_keypoints * size;

    double *cpu_patch_cost_gradient_hessian = new double[num_elems];
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> eigen_m(cpu_patch_cost_gradient_hessian, num_elems, 1);
    eigen_m.setRandom();

    double *gpu_patch_cost_gradient_hessian;
    cudaMalloc((void **)&gpu_patch_cost_gradient_hessian, sizeof(double) * num_elems);
    cudaMemcpy(gpu_patch_cost_gradient_hessian, cpu_patch_cost_gradient_hessian, sizeof(double) * num_elems, cudaMemcpyHostToDevice);

    // compute GPU results
    double *gpu_frame_cost_gradient_hessian;
    cudaMalloc((void **)&gpu_frame_cost_gradient_hessian, sizeof(double) * num_frames * 325);

    compute_frame_cost_gradient_hessian(num_frames, num_keypoints, 4, gpu_patch_cost_gradient_hessian, true, nullptr, gpu_frame_cost_gradient_hessian);

    double *cpu_frame_cost_gradient_hessian = new double[num_frames * 325];

    cudaMemcpy(cpu_frame_cost_gradient_hessian, gpu_frame_cost_gradient_hessian, sizeof(double) * num_frames * 325, cudaMemcpyDeviceToHost);

    // compute CPU results
    const int frame_idx = random_int(0, num_frames);
    Eigen::Matrix<double, 325, 1> cpu_sum;
    cpu_sum.setZero();
    for (int i = 0; i < num_keypoints; i++)
    {
        Eigen::Matrix<double, 325, 1> J(cpu_patch_cost_gradient_hessian + (frame_idx * num_keypoints + i) * 325);
        cpu_sum += J;
    }

    Eigen::Matrix<double, 325, 1> gpu_sum(cpu_frame_cost_gradient_hessian + frame_idx * 325);
    double max_err = (cpu_sum - gpu_sum).cwiseAbs().maxCoeff();

    std::cout << "max err: " << max_err << "\n";
    if (max_err > 1e-8)
    {
        std::cout << "cpu_sum: " << cpu_sum.transpose() << "\n";
        std::cout << "gpu_sum: " << gpu_sum.transpose() << "\n";
    }
}

void test_merge_hessian_gradient_cost()
{
    std::cout << "\n---------------" << __FUNCTION__ << "----------------\n";
    const int num_frames = 3;
    const int num_keypoints = 145;
    const int patch_size = 8;
    const int num_pixels = num_frames * num_keypoints * patch_size;
    const int num_patches = num_frames * num_keypoints;

    double *cpu_pixel_residuals = new double[num_pixels];
    double *cpu_pixel_jacobians = new double[num_pixels * 24];
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>
        eigen_cpu_pixel_residuals(cpu_pixel_residuals, num_pixels);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>>
        eigen_cpu_pixel_jacobians(cpu_pixel_jacobians, num_pixels * 24);
    eigen_cpu_pixel_residuals.setRandom();
    eigen_cpu_pixel_jacobians.setRandom();

    double *gpu_pixel_residuals;
    double *gpu_pixel_jacobians;
    cudaMalloc((void **)&gpu_pixel_residuals, sizeof(double) * num_pixels);
    cudaMalloc((void **)&gpu_pixel_jacobians, sizeof(double) * num_pixels * 24);
    cudaMemcpy(gpu_pixel_residuals, cpu_pixel_residuals, sizeof(double) * num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_pixel_jacobians, cpu_pixel_jacobians, sizeof(double) * num_pixels * 24, cudaMemcpyHostToDevice);

    // compute cost, gradient, hessian with GPU
    double *gpu_patch_cost_gradient_hessian;
    cudaMalloc((void **)&gpu_patch_cost_gradient_hessian, sizeof(double) * num_patches * 325);

    compute_patch_cost_gradient_hessian(num_frames,
                                        num_keypoints,
                                        patch_size,
                                        4,
                                        gpu_pixel_residuals,
                                        gpu_pixel_jacobians,
                                        1e32,
                                        1,
                                        gpu_patch_cost_gradient_hessian);

    double *gpu_frame_cost_gradient_hessian;
    cudaMalloc((void **)&gpu_frame_cost_gradient_hessian, sizeof(double) * num_frames * 325);

    compute_frame_cost_gradient_hessian(num_frames, num_keypoints, 4, gpu_patch_cost_gradient_hessian, true, nullptr, gpu_frame_cost_gradient_hessian);

    //
    const int num_ctrl_knots = num_frames + 3;
    std::vector<int> ctrl_knot_start_indices;
    for (int i = 0; i < num_frames; i++)
    {
        ctrl_knot_start_indices.push_back(i);
    }
    const int ctrl_knot_H_W = num_ctrl_knots * 6;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Hgpu, Hcpu, bgpu, bcpu;
    Hgpu.resize(ctrl_knot_H_W, ctrl_knot_H_W);
    Hcpu.resize(ctrl_knot_H_W, ctrl_knot_H_W);
    bgpu.resize(ctrl_knot_H_W, 1);
    bcpu.resize(ctrl_knot_H_W, 1);
    double total_cost_gpu = 0;
    double total_cost_cpu = 0;

    merge_hessian_gradient_cost(num_frames,
                                4,
                                gpu_frame_cost_gradient_hessian,
                                ctrl_knot_start_indices.data(),
                                num_ctrl_knots,
                                &total_cost_gpu,
                                Hgpu.data(),
                                bgpu.data());

    // compute CPU results
    Hcpu.setZero();
    bcpu.setZero();
    for (int i = 0; i < num_pixels; i++)
    {
        double r = cpu_pixel_residuals[i];
        Eigen::Matrix<double, 24, 1> J(cpu_pixel_jacobians + i * 24);

        int frame_idx = i / (num_keypoints * patch_size);
        int start_idx = ctrl_knot_start_indices.at(frame_idx);

        total_cost_cpu += 0.5 * r * r;

        Eigen::Matrix<double, 24, 1> b = r * J;
        Eigen::Matrix<double, 24, 24> H = J * J.transpose();

        bcpu.block(start_idx * 3, 0, 12, 1) += b.head(12);
        bcpu.block((num_ctrl_knots + start_idx) * 3, 0, 12, 1) += b.tail(12);

        Hcpu.block(start_idx * 3, start_idx * 3, 12, 12) += H.block(0, 0, 12, 12);
        Hcpu.block((num_ctrl_knots + start_idx) * 3, start_idx * 3, 12, 12) += H.block(12, 0, 12, 12);
        Hcpu.block(start_idx * 3, (num_ctrl_knots + start_idx) * 3, 12, 12) += H.block(0, 12, 12, 12);
        Hcpu.block((num_ctrl_knots + start_idx) * 3, (num_ctrl_knots + start_idx) * 3, 12, 12) += H.block(12, 12, 12, 12);
    }

    double max_cost_err = fabs(total_cost_cpu - total_cost_gpu);
    double max_grad_err = (bgpu - bcpu).cwiseAbs().maxCoeff();
    double max_hess_err = (Hgpu - Hcpu).cwiseAbs().maxCoeff();

    std::cout << "max_cost_err: " << max_cost_err << "\n";
    if (max_cost_err > 1e-4)
    {
        std::cout << "total_cost_gpu: " << total_cost_gpu << "\n";
        std::cout << "total_cost_cpu: " << total_cost_cpu << "\n\n";
    }

    std::cout << "max_grad_err: " << max_grad_err << "\n";
    if (max_grad_err > 1e-4)
    {
        std::cout << "bcpu: " << bcpu.transpose() << "\n\n";
        std::cout << "bgpu: " << bgpu.transpose() << "\n\n";
    }

    std::cout << "max_hess_err: " << max_hess_err << "\n";
    if (max_hess_err > 1e-4)
    {
        std::cout << "Hcpu: \n"
                  << Hcpu.block(0, 12, 12, 12) << "\n\n";
        std::cout << "Hgpu: \n"
                  << Hgpu.block(0, 12, 12, 12) << "\n\n";
    }
}

void test_solve_normal_equation()
{
#define N 48
    std::cout << "\n---------------" << __FUNCTION__ << "----------------\n";
    Eigen::Matrix<double, N, N> A;
    Eigen::Matrix<double, N, 1> b, x, x_;

    A.setRandom();
    A = A.transpose() * A;

    x.setRandom();
    b = A * (-x);

    WallTime t_start = currentTime();
    solve_normal_equation(A, b, 1, x_);
    std::cout << "solve linear system of equations with size " << N << " consumes " << elapsedTimeInMicroSeconds(t_start) << " us...\n";

    double max_err = (x - x_).cwiseAbs().maxCoeff();
    std::cout << "max_err: " << max_err << "\n";
    if (max_err > 1e-8)
    {
        std::cout << "solution: " << x.transpose() << "\n";
        std::cout << "solved  : " << x_.transpose() << "\n";
    }
#undef N
}

int main(int argc, char **argv)
{
    // srand(time(nullptr));
    test_compute_pixel_intensity();
    test_compute_virtual_camera_poses();
    test_compute_local_patches();
    test_compute_pixel_jacobian_residual();
    test_compute_patch_cost_gradient_hessian();
    test_compute_frame_cost_gradient_hessian();
    test_merge_hessian_gradient_cost();
    test_solve_normal_equation();
    return 0;
}