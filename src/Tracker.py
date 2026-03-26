# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Tracker.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

from __future__ import annotations

from colorama import Fore, Style

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from src.common import matrix_to_cam_pose, cam_pose_to_matrix, get_samples, get_samples_blur

### spline
from src.utils.Spline import noise_T


if TYPE_CHECKING:
    import argparse
    from src.ESLAM import ESLAM


class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg: dict, args: argparse.Namespace, eslam: ESLAM):
        self.cfg = cfg
        self.args = args
        self.eslam = eslam

        self.scale = cfg['scale']

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

        # for p in self.decoders.parameters():
        #     p.requires_grad_(False)

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.eslam.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.eslam.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.eslam.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.eslam.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.eslam.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.eslam.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def optimize_tracking(self, all_planes, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.eslam.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth
            inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        depth, color, sdf, z_vals = self.eslam.renderer.render_batch_ray(
            all_planes, self.eslam.decoders, batch_rays_d, batch_rays_o,
            self.device, self.eslam.truncation, gt_depth=batch_gt_depth)

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)

        ## SDF losses
        loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

        ## Color Loss
        loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()

        ### Depth loss
        loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    ### here, idx >= 1?
    def predict_current_pose(self, idx):
        if self.const_speed_assumption and idx - 2 >= 0:
            ## Linear prediction for initialization  ### here
            # delta = self.eslam.estimate_c2ws[idx - 2].inverse() @ self.eslam.pre_c2w
            # cam_pose = matrix_to_cam_pose((self.eslam.pre_c2w @ delta)[None, ...])
            pre_poses = torch.stack([self.eslam.estimate_c2ws[idx - 2], self.eslam.estimate_c2ws[idx - 1]], dim=0)
            pre_poses = matrix_to_cam_pose(pre_poses)
            cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]
        else:
            ## Initialize with the last known pose
            cam_pose = matrix_to_cam_pose(self.eslam.estimate_c2ws[idx - 1][None, ...])
        return cam_pose

    def tracking_render(self, batch):
        idx = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        gt_color = gt_color.to(self.device, non_blocking=True)
        gt_depth = gt_depth.to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.to(self.device, non_blocking=True)

        # self.eslam.decoders.eval()
        all_planes = self.eslam.get_all_planes()  ### here

        if self.eslam.verbose:
            print(Fore.MAGENTA)
            print("Tracking Frame ", idx)
            print(Style.RESET_ALL)

        # # update keyframe pose from BA, replace pre_c2w with self.eslam.estimate_c2ws[idx - 1]
        # if idx > 0 and (idx % self.eslam.keyframe_every == 1 or self.eslam.keyframe_every == 1):
        #     self.eslam.pre_c2w = self.eslam.estimate_c2ws[idx - 1].to(self.device)

        cam_pose = self.predict_current_pose(idx)
        if not self.gt_camera:
            T = torch.nn.Parameter(cam_pose[:, -3:].clone())
            R = torch.nn.Parameter(cam_pose[:, :4].clone())
            cam_para_list_T = [T]
            cam_para_list_R = [R]
            cam_optimizer = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas': (0.5, 0.999)},
                                              {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas': (0.5, 0.999)}])

            current_min_loss = torch.tensor(float('inf')).float().to(self.device)
            for cam_iter in range(self.num_cam_iters):
                cam_pose = torch.cat([R, T], -1)

                self.eslam.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose,
                                                all_planes, self.eslam.decoders)

                loss = self.optimize_tracking(all_planes, cam_pose, gt_color, gt_depth, self.tracking_pixels, cam_optimizer)
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_pose = cam_pose.clone().detach()

            c2w = cam_pose_to_matrix(candidate_cam_pose)
        else:
            c2w = gt_c2w

        self.eslam.estimate_c2ws[idx] = c2w.squeeze(0).clone()
        self.eslam.gt_c2ws[idx] = gt_c2w.squeeze(0).clone().cpu()
        # self.eslam.pre_c2w = c2w.squeeze(0).clone()

        # save relative poses of non-keyframes to keyframe
        if idx % self.eslam.keyframe_every != 0:
            kf_id = idx // self.eslam.keyframe_every
            kf_frame_id = kf_id * self.eslam.keyframe_every
            c2w_key = self.eslam.estimate_c2ws[kf_frame_id]
            delta = self.eslam.estimate_c2ws[idx] @ c2w_key.float().inverse()
            self.eslam.estimate_c2ws_rel[idx] = delta

        if self.eslam.verbose:
            print('Best loss: {}, Camera loss{}'.format(
                F.l1_loss(c2w[0, :3], gt_c2w[0, :3]).cpu().item(),
                F.l1_loss(cam_pose_to_matrix(cam_pose.clone().detach())[0, :3], gt_c2w[0, :3]).cpu().item()))

        # self.eslam.decoders.train()

    def tracking_render_blur_gtcamera(self, batch):
        idx = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        gt_color = gt_color.to(self.device, non_blocking=True)
        gt_depth = gt_depth.to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.to(self.device, non_blocking=True)

        # self.eslam.decoders.eval()
        all_planes = self.eslam.get_all_planes()  ### here

        if self.eslam.verbose:
            print(Fore.MAGENTA)
            print("Tracking Frame ", idx)
            print(Style.RESET_ALL)

        # # update keyframe pose from BA, replace pre_c2w with self.eslam.estimate_c2ws[idx - 1]
        # if idx > 0 and (idx % self.eslam.keyframe_every == 1 or self.eslam.keyframe_every == 1):
        #     self.eslam.pre_c2w = self.eslam.estimate_c2ws[idx - 1].to(self.device)

        cam_pose = self.predict_current_pose(idx)
        c2w = gt_c2w
        c2w_start = gt_c2w @ noise_T()
        c2w_end = gt_c2w @ noise_T()

        self.eslam.estimate_c2ws[idx] = c2w.squeeze(0).clone()
        self.eslam.estimate_c2ws_start[idx] = c2w_start.squeeze(0).clone()
        self.eslam.estimate_c2ws_end[idx] = c2w_end.squeeze(0).clone()
        self.eslam.gt_c2ws[idx] = gt_c2w.squeeze(0).clone().cpu()
        # self.eslam.pre_c2w = c2w.squeeze(0).clone()

        # save relative poses of non-keyframes to keyframe
        if idx % self.eslam.keyframe_every != 0:
            kf_id = idx // self.eslam.keyframe_every
            kf_frame_id = kf_id * self.eslam.keyframe_every
            c2w_key = self.eslam.estimate_c2ws[kf_frame_id]
            delta = self.eslam.estimate_c2ws[idx] @ c2w_key.float().inverse()
            self.eslam.estimate_c2ws_rel[idx] = delta

        if self.eslam.verbose:
            print('Best loss: {}, Camera loss{}'.format(
                F.l1_loss(c2w[0, :3], gt_c2w[0, :3]).cpu().item(),
                F.l1_loss(cam_pose_to_matrix(cam_pose.clone().detach())[0, :3], gt_c2w[0, :3]).cpu().item()))

        # self.eslam.decoders.train()