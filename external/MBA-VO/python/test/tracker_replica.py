import json
from pathlib import Path
import glob

import cv2
import numpy as np
import pypose as pp
import torch
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
from pymbavo import MBAVO


PROJECT_ROOT = Path("/home/lzzhao/ws/Co-SLAM/external/MBA-VO")
DATASET_ROOT = Path("/home/lzzhao/ws/Co-SLAM/data/Replica/office0/")
CONFIG_JSON = PROJECT_ROOT / "test" / "blur_aware_tracker_configs_replica.json"

PNG_DEPTH_SCALE = 6553.5

def load_images(path_to_sequence: Path):
    rgb_images = sorted(glob.glob(str(path_to_sequence / "*.jpg")))
    depth_images = sorted(glob.glob(str(path_to_sequence / "*.png")))
    assert len(rgb_images) == len(depth_images)
    return {
        "rgb": rgb_images,
        "depth": depth_images,
    }


if __name__ == "__main__":
    with open(CONFIG_JSON, 'r') as f:
        config = json.load(f)["Configuration"]
    vo = MBAVO(config)

    print(f"==========tracker configs==========\n"
          f"{vo.tracker.get_configs()}\n")

    images = load_images(Path(DATASET_ROOT) / "results")
    num_images = len(images["rgb"])

    print(f'config["im_ref_idx"]: {config["im_ref_idx"]}')

    exposure_time = config["exposure_time"]

    with open("results_python.txt", "w") as f:
        f.write("#timestamp x y z qx qy qz qw\n")
        last_kf_pose = pp.identity_SE3(1)

        # iterate over the dict images
        for i in range(config["im_ref_idx"], num_images-1):
            print(f"==========[{i}]==========")

            im_ref_idx = i
            im_cur_idx = i + 1
            timestamp_ref = float(im_ref_idx)
            timestamp_cur = float(im_cur_idx)

            im_cur_path = images['rgb'][im_cur_idx]
            im_ref_path = images['rgb'][im_ref_idx]
            im_depth_path = images['depth'][im_ref_idx]

            im_cur = cv2.imread(im_cur_path, cv2.IMREAD_GRAYSCALE)
            im_ref = cv2.imread(im_ref_path, cv2.IMREAD_GRAYSCALE)
            im_depth = cv2.imread(im_depth_path, cv2.IMREAD_ANYDEPTH)

            im_cur_tensor = torch.tensor(im_cur)[:, :, None]
            im_ref_tensor = torch.tensor(im_ref)[:, :, None]
            im_depth_tensor = torch.tensor(im_depth.astype(np.float32) / PNG_DEPTH_SCALE)
            if im_depth_tensor.dim() == 3:
                im_depth_tensor = im_depth_tensor[:, :, 0]
            im_depth_tensor = im_depth_tensor[:, :, None].contiguous()

            print(im_cur.shape)
            print(im_ref.shape)
            print('\n')
            print(im_cur_tensor.shape)
            print(im_ref_tensor.shape)
            print(im_depth_tensor.shape)

            if i == config["im_ref_idx"]:
                vo.tracker.set_sharp_frame_from_tensor(im_ref_idx, im_ref_tensor, timestamp_ref)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()
                vo.tracker.temp_process_keyframe()

            vo.tracker.set_blur_frame_from_tensor(im_cur_idx, im_cur_tensor, timestamp_cur, exposure_time)

            # is_keyframe = vo.tracker.track_coarse()
            is_keyframe = vo.tracker.track()
            vo.tracker.eval_poses(is_keyframe)

            t = vo.tracker.get_translation()
            q = vo.tracker.get_rotation()
            delta = pp.SE3(torch.cat([torch.from_numpy(t), torch.from_numpy(q)], dim=-1)).float()
            cur_est = last_kf_pose @ delta
            if is_keyframe:
                last_kf_pose = cur_est
                vo.tracker.set_sharp_frame_from_tensor(im_cur_idx, im_cur_tensor, timestamp_cur)
                vo.tracker.set_depth_image_from_tensor(im_depth_tensor)
                vo.tracker.set_keyframe()
                vo.tracker.temp_process_keyframe()

            cur_est_str = ' '.join([str(x) for x in cur_est.numpy().flatten()])
            print(cur_est_str)
            f.write(f"{timestamp_cur} {cur_est_str}\n")
