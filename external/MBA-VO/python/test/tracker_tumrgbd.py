import os
import json
from pathlib import Path
import glob

import cv2
import numpy as np
import pypose as pp
import torch
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
from pymbavo import MBAVO


torch.set_default_dtype(torch.float64)

PROJECT_ROOT = Path("/home/lzzhao/ws/Co-SLAM/external/MBA-VO")
DATASET_ROOT = Path("/home/lzzhao/ws/Co-SLAM/data/TUM/rgbd_dataset_freiburg1_desk")
CONFIG_JSON = PROJECT_ROOT / "test" / "blur_aware_tracker_configs_tumrgbd.json"

PNG_DEPTH_SCALE = 5000.0

def load_images(path_to_sequence: Path):
    rgb_images = sorted(glob.glob(str(path_to_sequence / "rgb" /"*.png")))
    depth_images = sorted(glob.glob(str(path_to_sequence / "depth" / "*.png")))

    # align rgb and depth images with timestamps
    rgb_images_aligned = []
    depth_images_aligned = []
    for rgb in rgb_images:
        rgb_timestamp = float(os.path.basename(rgb)[0:-4])
        for depth in depth_images:
            depth_timestamp = float(os.path.basename(depth)[0:-4])
            if abs(rgb_timestamp - depth_timestamp) < 0.02:
                rgb_images_aligned.append(rgb)
                depth_images_aligned.append(depth)
                print(f"rgb: {rgb_timestamp}, depth: {depth_timestamp}")
                print(f"delta: {rgb_timestamp - depth_timestamp}")
                break

    assert len(rgb_images_aligned) == len(depth_images_aligned)
    return {
        "rgb": rgb_images_aligned,
        "depth": depth_images_aligned,
    }


if __name__ == "__main__":
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)["Configuration"]
    vo = MBAVO(config)

    print(f"==========tracker configs==========\n"
          f"{vo.tracker.get_configs()}\n")

    images = load_images(Path(DATASET_ROOT))
    num_images = len(images["rgb"])
    print(f"num_images: {num_images}")
    print(f'config["im_ref_idx"]: {config["im_ref_idx"]}')

    exposure_time = config["exposure_time"]

    with open("results_python.txt", "w") as f:
        f.write("#timestamp x y z qx qy qz qw\n")
        last_kf_pose = pp.identity_SE3(1)

        # iterate over the dict images
        for i in range(config["im_ref_idx"], num_images):
            print(f"==========[{i}]==========")

            timestamp = float(os.path.basename(images['rgb'][i])[0:-4])
            # timestamp = float(i)
            print(f"timestamp: {timestamp}")

            im_path = images['rgb'][i]
            im_depth_path = images['depth'][i]

            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            im_depth = cv2.imread(im_depth_path, cv2.IMREAD_ANYDEPTH)

            im_tensor = torch.tensor(im)[:, :, None]
            im_depth_tensor = torch.tensor(im_depth.astype(np.float32) / PNG_DEPTH_SCALE)
            if im_depth_tensor.dim() == 3:
                im_depth_tensor = im_depth_tensor[:, :, 0]
            im_depth_tensor = im_depth_tensor[:, :, None].contiguous()

            print(im.shape)
            print('\n')
            print(im_tensor.shape)
            print(im_depth_tensor.shape)

            if i == config["im_ref_idx"]:
                vo.tracker.set_sharp_frame_from_tensor(i, im_tensor, timestamp)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()
                continue

            vo.tracker.set_blur_frame_from_tensor(i, im_tensor, timestamp, exposure_time)

            is_keyframe = vo.tracker.track_coarse()
            vo.tracker.eval_poses(is_keyframe)

            t = vo.tracker.get_translation()
            q = vo.tracker.get_rotation()
            delta = pp.SE3(torch.cat([torch.from_numpy(t), torch.from_numpy(q)], dim=-1))
            cur_est = last_kf_pose @ delta
            if is_keyframe:
                last_kf_pose = cur_est
                vo.tracker.set_sharp_frame_from_tensor(i, im_tensor, timestamp)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()

            cur_est_str = ' '.join([str(x) for x in cur_est.numpy().flatten()])
            print(cur_est_str)
            f.write(f"{timestamp} {cur_est_str}\n")
