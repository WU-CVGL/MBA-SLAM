import json
import cv2
import numpy as np
import torch

import pymbavo

config_json_path = "/home/lzzhao/ws/Co-SLAM/external/MBA-VO/test/blur_aware_tracker_configs_unreal_new.json"
depth_image_path  = "/home/lzzhao/data/blur/ArchVizInterior_eth3d_camera_shake_1_trajectory/depth_sharp_middle/4089.5900.exr"
rgb_image_path = "/home/lzzhao/data/blur/ArchVizInterior_eth3d_camera_shake_1_trajectory/rgb_blur/4089.5900.jpg"

print("==========config_json==========")
with open(config_json_path, 'r') as f:
    cfg = json.load(f)["Configuration"]
print(cfg)

print("==========Vector2d==========")
a = pymbavo.Vector2d(1,3.14)
print(a(0), a(1))

print("==========PinholeCamera==========")
K = np.array([320,320,320,240])
cam = pymbavo.PinholeCamera(K, 640, 480)

print("==========MBA-VO tracker constructor==========")
vo = pymbavo.MBAVO(cfg)
vo.tracker.show()

print("==========UnrealDepthImageCpu==========")
a = vo.tracker.GetUnrealDepthImageCpu(depth_image_path)
print(a.shape)

print("==========UnrealDepthImageGpu==========")
b = vo.tracker.GetUnrealDepthImageGpu(depth_image_path)
print(b.shape)

print("==========Image && Pyramid && Frame==========")
img0 = cv2.imread(rgb_image_path)
img1 = cv2.resize(img0, (0,0), fx=0.5, fy=0.5)
img2 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
img3 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
img0 = torch.from_numpy(img0.astype(np.float32) / 255)
img1 = torch.from_numpy(img1.astype(np.float32) / 255)
img2 = torch.from_numpy(img2.astype(np.float32) / 255)
img3 = torch.from_numpy(img3.astype(np.float32) / 255)
print(img0.shape)
print(img1.shape)
print(img2.shape)
print(img3.shape)
vo_img0 = vo.factory.get_imageu8_from_tensor(img0)
vo_img1 = vo.factory.get_imageu8_from_tensor(img1)
vo_img2 = vo.factory.get_imageu8_from_tensor(img2)
vo_img3 = vo.factory.get_imageu8_from_tensor(img3)
vo_pyramid = pymbavo.ImagePyramidU8()
vo_pyramid.set_pyramid([vo_img0, vo_img1, vo_img2, vo_img3])

frame = pymbavo.Frame(0)
# frame.insert_image(0, img0)
# frame.compute_image_pyramid(0, 1)
frame.set_image_pyramid(0, vo_pyramid)
frame.compute_grad_image_pyramid(0, 1)

print("==========detector_cfgs==========")
detector_cfgs = vo.tracker.get_detector_configs_copy()
# print(detector_cfgs.detector_type)
print(f"score_threshold: {detector_cfgs.score_threshold}")
print(f"grid_selection_cell_H: {detector_cfgs.grid_selection_cell_H}")
print(f"grid_selection_cell_W: {detector_cfgs.grid_selection_cell_W}")
detector_cfgs_copy = vo.tracker.get_detector_configs_copy()
detector_cfgs_copy.grid_selection_cell_H *= 2
detector_cfgs_copy.grid_selection_cell_W *= 2
print(f"COPIED grid_selection_cell_H: {detector_cfgs_copy.grid_selection_cell_H}")
print(f"COPIED grid_selection_cell_W: {detector_cfgs_copy.grid_selection_cell_W}")
print(f"ORIGINAL grid_selection_cell_H: {detector_cfgs.grid_selection_cell_H}")
print(f"ORIGINAL grid_selection_cell_W: {detector_cfgs.grid_selection_cell_W}")
frame.detect_features(0, detector_cfgs)
keypoints = vo.factory.get_keypoints_from_frame(frame, 0, 0)
print(f"Detected keyponts: {keypoints.shape}")
print(keypoints[0])
