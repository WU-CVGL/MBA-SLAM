import os
import cv2
import torch
import json
import typing
import pypose as pp
import torch
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
from pymbavo import MBAVO
from frame import Frame, FrameConfig
from detectors import ImagePyramidGridDetectorConfig


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
torch.set_default_dtype(torch.float64)

PROJECT_ROOT = "/home/lzzhao/ws/Co-SLAM/external/MBA-VO"
DATASET_ROOT = "/home/lzzhao/data/blur-slam/tanabata_blur"
# DATASET_ROOT = "/home/lzzhao/data/blur-slam/tanabata_blur/tanabata_easy_full_bad_nerf"
GT_ROOT = "/home/lzzhao/data/blur-slam/tanabata_blur"
CONFIG_JSON = os.path.join(PROJECT_ROOT, "test/blur_aware_tracker_configs_blender.json")
TIMESTAMPS_TXT = os.path.join(DATASET_ROOT, "timestamps.txt")

DEBUG_USE_GT_RGB_FRAME_ID = 999
DEBUG_USE_GT_DEPTH_FRAME_ID = 999

def load_image_list(path_to_list: os.PathLike) -> typing.List[str]:
    with open(path_to_list, 'r') as f:
        buffer = f.read().splitlines()
    im_list = []
    for line in buffer:
        line = os.path.splitext(line)[0]
        im_list.append(line)
    return im_list


if __name__ == "__main__":
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)["Configuration"]
    vo = MBAVO(config)

    num_pyramid_levels = config["num_pyramid_level"]
    frame_config: FrameConfig = FrameConfig(
        patch_size=9,
        detector=ImagePyramidGridDetectorConfig(
            num_pyramid_levels=num_pyramid_levels,
            max_num_keypoints=1000,
        ),
    )
    # blur_frame: Frame = self.tracker_state.config.frame.setup(device=self.device)
    sharp_frame: Frame = frame_config.setup(device="cpu")
    depth_frame: Frame = frame_config.setup(device="cpu")

    print(f"==========tracker configs==========\n"
          f"{vo.tracker.get_configs()}\n")

    blur_dir = os.path.join(DATASET_ROOT, config["rel_path_to_curr_ims"])
    sharp_dir = os.path.join(DATASET_ROOT, config["rel_path_to_ref_ims"])
    depth_dir = os.path.join(DATASET_ROOT, config["rel_path_to_depth_ims"])
    ext = config["im_ext"]

    gt_sharp_dir = os.path.join(GT_ROOT, config["rel_path_to_ref_ims"])
    gt_depth_dir = os.path.join(GT_ROOT, config["rel_path_to_depth_ims"])

    im_list_path = os.path.join(DATASET_ROOT, config["rel_path_to_im_list"])
    im_list = load_image_list(im_list_path)
    print(f'config["im_ref_idx"]: {config["im_ref_idx"]}')

    exposure_time = config["exposure_time"]

    timestamps = []
    with open(TIMESTAMPS_TXT, "r") as f:
        for line in f.read().splitlines():
            timestamps.append(float(line))

    with open("results_python.txt", "w") as f:
        f.write("#timestamp x y z qx qy qz qw\n")
        last_kf_pose = pp.identity_SE3(1)
        for i in range(config["im_ref_idx"], len(im_list)):
            capture_time = timestamps[i]
            print(f"=========={i}: {capture_time}==========")

            im_blur_path = os.path.join(blur_dir, im_list[i] + ext)
            im_sharp_path = os.path.join(sharp_dir, im_list[i] + ext)
            im_depth_path = os.path.join(depth_dir, im_list[i] + '.exr')

            if i > DEBUG_USE_GT_RGB_FRAME_ID:
                im_sharp_path = os.path.join(gt_sharp_dir, im_list[i] + ext)

            if i > DEBUG_USE_GT_DEPTH_FRAME_ID:
                im_depth_path = os.path.join(gt_depth_dir, im_list[i] + '.exr')

            im_blur = cv2.imread(im_blur_path, cv2.IMREAD_GRAYSCALE)
            im_sharp = cv2.imread(im_sharp_path, cv2.IMREAD_GRAYSCALE)
            im_depth = cv2.imread(im_depth_path, cv2.IMREAD_UNCHANGED)

            im_blur_tensor = torch.tensor(im_blur)[:, :, None]
            im_sharp_tensor = torch.tensor(im_sharp)[:, :, None]
            im_depth_tensor = torch.tensor(im_depth)
            if im_depth_tensor.dim() == 3:
                im_depth_tensor = im_depth_tensor[:, :, 0]
            im_depth_tensor = im_depth_tensor[:, :, None].contiguous()

            print(im_blur.shape)
            print(im_sharp.shape)
            print(im_depth.shape)
            print('\n')
            print(im_blur_tensor.shape)
            print(im_sharp_tensor.shape)
            print(im_depth_tensor.shape)

            vo.tracker.set_blur_frame_from_tensor(i, im_blur_tensor, capture_time, exposure_time)
            if i == config["im_ref_idx"]:
                vo.tracker.set_sharp_frame_from_tensor(i, im_sharp_tensor, capture_time)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()
                vo.tracker.temp_process_keyframe()

            # is_keyframe = vo.tracker.track_coarse()
            is_keyframe = vo.tracker.track()
            vo.tracker.eval_poses(is_keyframe)

            t = vo.tracker.get_translation()
            q = vo.tracker.get_rotation()
            delta = pp.SE3(torch.cat([torch.from_numpy(t), torch.from_numpy(q)], dim=-1))
            cur_est = last_kf_pose @ delta
            if is_keyframe:
                last_kf_pose = cur_est
                vo.tracker.set_sharp_frame_from_tensor(i, im_sharp_tensor, capture_time)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()
                vo.tracker.temp_process_keyframe()

            cur_est_str = ' '.join([str(x) for x in cur_est.numpy().flatten()])
            print(cur_est_str)
            f.write(f"{capture_time} {cur_est_str}\n")
