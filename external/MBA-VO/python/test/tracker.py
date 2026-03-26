# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import pypose as pp
import torch
import torch.nn.functional as TF
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor

import pymbavo
from pymbavo import MBAVO


PROJECT_ROOT = "/home/lzzhao/ws/Co-SLAM/external/MBA-VO"
DATASET_ROOT = "/home/lzzhao/data/blur/ArchVizInterior_eth3d_camera_shake_1_trajectory"
# CONFIG_JSON = os.path.join(PROJECT_ROOT, "test/blur_aware_tracker_configs_unreal_new.json")

# MBA-VO sharp frames managed by python
vo_sharp_frames = {}


def get_pose(
        timestamp: float,
        gt_timestamps: Float[Tensor, "num_poses"],
        gt_poses: Float[LieTensor, "num_poses 7"]
) -> Float[LieTensor, "7"]:
    """Get pose from ground truth"""
    idx = torch.argmin(torch.abs(gt_timestamps - timestamp))
    assert torch.abs(gt_timestamps[idx] - timestamp) < 0.0099
    return gt_poses[idx]


def load_tum_trajectory(
        filename: Path
) -> Tuple[
    Float[Tensor, "num_poses"],
    Float[LieTensor, "num_poses 7"]
]:
    """Load TUM trajectory from file"""
    with open(filename, 'r', encoding="UTF-8") as f:
        lines = f.read().splitlines()
    if lines[0].startswith('#'):
        lines.pop(0)
    lines = [line.split() for line in lines]
    lines = [[float(val) for val in line] for line in lines]
    timestamps = []
    poses = []
    for line in lines:
        timestamps.append(torch.tensor(line[0]))
        poses.append(torch.tensor(line[1:]))
    timestamps = torch.stack(timestamps)
    poses = pp.SE3(torch.stack(poses))
    return timestamps, poses


def load_image_list(path_to_list: os.PathLike,
                    num_digits_for_im_name: int) -> List[str]:
    with open(path_to_list, 'r') as f:
        buffer = f.read().splitlines()
    im_list = []
    for line in buffer:
        line = line.split('#')[0]
        line = line.split(' ')[0]
        digit_pos = line.find('.')
        num_real_digits = len(line) - digit_pos - 1
        if num_real_digits != num_digits_for_im_name:
            num_chars_to_remove = num_real_digits - num_digits_for_im_name
            line = line[:-num_chars_to_remove]
        im_list.append(line)
    return im_list


def update_keyframe(
        tracker,
        i: int,
        capture_time: float,
        sharp_tensor: Tensor,
        depth_tensor: Tensor,
        num_pyramid_levels: int = 5
):
    # downscale sharp_tensor
    vo_pyramid_images = []
    last_image = sharp_tensor.squeeze()[None, None, ...]
    for _ in range(num_pyramid_levels):
        vo_image = pymbavo.Factory.get_imageu8_from_tensor(last_image.squeeze()[..., None])
        vo_pyramid_images.append(vo_image)
        last_image = TF.interpolate(last_image, scale_factor=0.5, mode="bilinear", align_corners=False)
    vo_pyramid = pymbavo.ImagePyramidU8()
    vo_pyramid.set_pyramid(vo_pyramid_images)

    vo_sharp_frames[i] = pymbavo.Frame(i)
    vo_sharp_frames[i].set_image_pyramid(0, vo_pyramid)
    vo_sharp_frames[i].set_capture_time(capture_time)
    vo_sharp_frames[i].compute_grad_image_pyramid(0, num_pyramid_levels)

    tracker.set_sharp_frame(vo_sharp_frames[i])
    # tracker.set_sharp_frame_from_tensor(i, sharp_tensor, capture_time)

    tracker.set_depth_image_from_tensor(depth_tensor)
    tracker.set_keyframe()
    tracker.temp_process_keyframe()
    for lvl in range(num_pyramid_levels):
        keypoints_xyz = tracker.get_keyframe_keypoints(lvl)
        print(f"Keypoints at level {lvl}: {keypoints_xyz.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config_json_path>")
        sys.exit(1)
    config_json_path = Path(sys.argv[1])
    assert config_json_path.exists(), f"{config_json_path} does not exist"
    with open(config_json_path, 'r') as f:
        config = json.load(f)["Configuration"]

    vo = MBAVO(config)

    print(f"==========tracker configs==========\n"
          f"{vo.tracker.get_configs()}\n")

    sharp_dir = os.path.join(DATASET_ROOT, config["rel_path_to_ref_ims"])
    blur_dir = os.path.join(DATASET_ROOT, config["rel_path_to_curr_ims"])
    # legacy depth
    # depth_dir = os.path.join(DATASET_ROOT, config["rel_path_to_depth_ims"])
    depth_dir = os.path.join(DATASET_ROOT, "depth_exr_middle")
    ext = config["im_ext"]

    im_list_path = os.path.join(DATASET_ROOT, config["rel_path_to_im_list"])
    im_list = load_image_list(im_list_path, config["num_digits_for_im_name"])
    print(f'config["im_ref_idx"]: {config["im_ref_idx"]}')

    gt_poses_path = os.path.join(DATASET_ROOT, config["rel_path_to_gt_poses"])
    gt_timestamps, gt_poses = load_tum_trajectory(gt_poses_path)

    exposure_time = config["exposure_time"]
    num_pyramid_levels = config["num_pyramid_level"]

    # initialize spline trajectory
    assert config["num_init_ctrl_knots"] == 2
    Rc2b = pp.SO3(torch.tensor([
        config["extrinsic_Rb2w_qx"],
        config["extrinsic_Rb2w_qy"],
        config["extrinsic_Rb2w_qz"],
        config["extrinsic_Rb2w_qw"]
    ]))
    ref_time = gt_timestamps[config["im_ref_idx"]]
    cur_start_time = gt_timestamps[config["im_ref_idx"] + 1] - 0.5 * exposure_time
    cur_end_time = cur_start_time + config["dt_ctrl_knot"]
    pose_ref = get_pose(ref_time, gt_timestamps, gt_poses)
    pose_cur_start = get_pose(cur_start_time, gt_timestamps, gt_poses)
    pose_cur_end = get_pose(cur_end_time, gt_timestamps, gt_poses)

    q_ref = (pose_ref.rotation() @ Rc2b).squeeze().cpu().numpy()
    q_cur_start = (pose_cur_start.rotation() @ Rc2b).squeeze().cpu().numpy()
    q_cur_end = (pose_cur_end.rotation() @ Rc2b).squeeze().cpu().numpy()
    t_ref = pose_ref.translation().squeeze().cpu().numpy()
    t_cur_start = pose_cur_start.translation().squeeze().cpu().numpy()
    t_cur_end = pose_cur_end.translation().squeeze().cpu().numpy()

    print(f"==========spline trajectory==========\n"
          f"ref_time: {ref_time}\n"
          f"cur_start_time: {cur_start_time}\n"
          f"cur_end_time: {cur_end_time}\n"
          f"q_ref: {q_ref}\n"
          f"q_cur_start: {q_cur_start}\n"
          f"q_cur_end: {q_cur_end}\n"
          f"t_ref: {t_ref}\n"
          f"t_cur_start: {t_cur_start}\n"
          f"t_cur_end: {t_cur_end}\n")

    vo.tracker.init_trajectory(cur_start_time, q_ref, q_cur_start, q_cur_end, t_ref, t_cur_start, t_cur_end)

    # start tracking
    input("Press Enter to start tracking...")
    with open("results_python.txt", "w") as f:
        f.write("#timestamp x y z qx qy qz qw\n")
        last_kf_pose = pp.identity_SE3(1)
        for i in range(config["im_ref_idx"], len(im_list)):
            print(f"=========={i}: {im_list[i]}==========")
            capture_time = float(im_list[i])

            im_blur_path = os.path.join(blur_dir, im_list[i] + ext)
            im_sharp_path = os.path.join(sharp_dir, im_list[i] + ext)
            im_depth_path = os.path.join(depth_dir, im_list[i] + '.exr')
            im_blur = cv2.imread(im_blur_path, cv2.IMREAD_GRAYSCALE)
            im_sharp = cv2.imread(im_sharp_path, cv2.IMREAD_GRAYSCALE)
            im_depth = cv2.imread(im_depth_path, cv2.IMREAD_UNCHANGED)

            im_blur_tensor = torch.tensor(im_blur)[:, :, None]
            im_sharp_tensor = torch.tensor(im_sharp)[:, :, None]
            im_depth_tensor = torch.tensor(im_depth)
            if im_depth_tensor.dim() == 3:
                im_depth_tensor = im_depth_tensor[:, :, 0]
            im_depth_tensor = im_depth_tensor[:, :, None].contiguous()

            # legacy depth
            # im_depth_path = os.path.join(depth_dir, im_list[i] + '.exr')
            # im_depth_tensor = vo.tracker.GetUnrealDepthImageCpu(im_depth_path)

            im_depth_cv2 = im_depth_tensor.numpy()

            print(im_blur_tensor.shape)
            print(im_sharp_tensor.shape)
            print(im_depth_tensor.shape)

            # First frame
            if i == config["im_ref_idx"]:
                update_keyframe(vo.tracker, i, capture_time, im_sharp_tensor, im_depth_tensor, num_pyramid_levels)
                cur_est_str = ' '.join([str(x) for x in pp.identity_SE3(1).numpy().flatten()])
                f.write(f"{capture_time} {cur_est_str}\n")
                continue

            vo.tracker.set_blur_frame_from_tensor(i, im_blur_tensor, capture_time, exposure_time)
            # is_keyframe = vo.tracker.track_coarse()
            is_keyframe = vo.tracker.track()
            vo.tracker.eval_poses(is_keyframe)

            t = vo.tracker.get_translation()
            q = vo.tracker.get_rotation()
            delta = pp.SE3(torch.cat([torch.from_numpy(t), torch.from_numpy(q)], dim=-1)).float()
            cur_est = last_kf_pose @ delta

            if is_keyframe:
                last_kf_pose = cur_est
                update_keyframe(vo.tracker, i, capture_time, im_sharp_tensor, im_depth_tensor, num_pyramid_levels)

            cur_est_str = ' '.join([str(x) for x in cur_est.numpy().flatten()])
            print(cur_est_str)
            f.write(f"{capture_time} {cur_est_str}\n")
