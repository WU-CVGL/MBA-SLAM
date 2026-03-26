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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/Mapper.py
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

import os
from colorama import Fore, Style

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from src.common import get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix, get_samples_blur
from src.utils.Frame_Visualizer import Frame_Visualizer

# spline
from src.utils.Spline import spline_linear_qt, return_blur_color_exclude_first, return_blur_color, return_mid_exclude_first, return_mid_continuous, return_mid

if TYPE_CHECKING:
    import argparse
    from src.ESLAM import ESLAM


class Mapper(object):
    """
    Mapping main class.
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
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']

        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr'] # The learning rate for camera poses during mapping
        self.mapping_pixels = cfg['mapping']['pixels']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.eslam.output, 'mapping_vis'), renderer=self.eslam.renderer,
                                           truncation=self.eslam.truncation, verbose=self.eslam.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy

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

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color: ground truth color image of the current frame.
            gt_depth: ground truth depth image of the current frame.
            c2w: camera to world matrix for target view (3x4 or 4x4 both fine).
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0),gt_color.unsqueeze(0), device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.eslam.estimate_c2ws[idx] for idx in self.eslam.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])     ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            # keyframe_dict (list): a list of dictionaries of keyframes info.
            # keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """
        all_planes = self.eslam.get_all_planes()
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(self.eslam.keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.eslam.keyframe_dict)-2, self.mapping_window_size-1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(self.eslam.keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(self.eslam.keyframe_list)-1] + [len(self.eslam.keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        ### here create_mapper_optimizer just once
        # decoders_para_list = []
        # decoders_para_list += list(self.eslam.decoders.parameters())
        #
        # planes_para = []
        # for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
        #     for i, plane in enumerate(planes):
        #         plane = nn.Parameter(plane)
        #         planes_para.append(plane)
        #         planes[i] = plane
        #
        # c_planes_para = []
        # for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
        #     for i, c_plane in enumerate(c_planes):
        #         c_plane = nn.Parameter(c_plane)
        #         c_planes_para.append(c_plane)
        #         c_planes[i] = c_plane

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(self.eslam.keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(self.eslam.keyframe_dict[frame]['color'].to(device))
                c2ws.append(self.eslam.keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(self.eslam.keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)

        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))

            camera_optimizer = torch.optim.Adam([{'params': [cam_poses], 'lr': self.joint_opt_cam_lr}])

        lr_factor_not_first = self.cfg['mapping']['lr_factor']
        self.eslam.map_optimizer.param_groups[0]['lr'] = self.cfg['mapping']['lr']['decoders_lr'] * lr_factor_not_first
        self.eslam.map_optimizer.param_groups[1]['lr'] = self.cfg['mapping']['lr']['planes_lr'] * lr_factor_not_first
        self.eslam.map_optimizer.param_groups[2]['lr'] = self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor_not_first

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.eslam.decoders)

            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device)

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.eslam.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            depth, color, sdf, z_vals = self.eslam.renderer.render_batch_ray(all_planes, self.eslam.decoders,batch_rays_d,
                                                                       batch_rays_o, device, self.eslam.truncation,
                                                                       gt_depth=batch_gt_depth)
            depth_mask = (batch_gt_depth > 0)

            ## SDF losses
            loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Color loss
            loss = loss + self.w_color * torch.square(batch_gt_color - color).mean()

            ### Depth loss
            loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            self.eslam.map_optimizer.zero_grad()
            if self.joint_opt:
                camera_optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.eslam.map_optimizer.step()
            if self.joint_opt:
                camera_optimizer.step()

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    self.eslam.keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w

    def optimize_mapping_blur(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, cur_c2w, cur_c2w_start, cur_c2w_end):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            # keyframe_dict (list): a list of dictionaries of keyframes info.
            # keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.
            cur_c2w_start (tensor): the estimated start camera to world matrix of current frame.
            cur_c2w_end (tensor): the estimated end camera to world matrix of current frame.

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
            cur_c2w_start: return the updated cur_c2w_start, return the same input cur_c2w_start if no joint_opt
            cur_c2w_end: return the updated cur_c2w_start, return the same input cur_c2w_end if no joint_opt
        """
        all_planes = self.eslam.get_all_planes()
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(self.eslam.keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.eslam.keyframe_dict)-2, self.mapping_window_size-1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)

        # add the last two keyframes and the current frame(use -1 to denote)
        if len(self.eslam.keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(self.eslam.keyframe_list)-1] + [len(self.eslam.keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        gt_depths = []
        gt_colors = []
        c2ws = []
        c2ws_start = []
        c2ws_end = []
        gt_c2ws = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(self.eslam.keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(self.eslam.keyframe_dict[frame]['color'].to(device))
                c2ws.append(self.eslam.keyframe_dict[frame]['est_c2w'])
                c2ws_start.append(self.eslam.keyframe_dict[frame]['est_c2w_start'])
                c2ws_end.append(self.eslam.keyframe_dict[frame]['est_c2w_end'])
                gt_c2ws.append(self.eslam.keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                c2ws.append(cur_c2w)
                c2ws_start.append(cur_c2w_start)
                c2ws_end.append(cur_c2w_end)
                gt_c2ws.append(gt_cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        c2ws_start = torch.stack(c2ws_start, dim=0)
        c2ws_end = torch.stack(c2ws_end, dim=0)

        kf_num = c2ws.shape[0] - 1
        if self.joint_opt:
            cam_poses_start = nn.Parameter(matrix_to_cam_pose(c2ws_start[1:]))  # wxyz
            cam_poses_end = nn.Parameter(matrix_to_cam_pose(c2ws_end[1:]))

            camera_optimizer = torch.optim.Adam([{'params': [cam_poses_start], 'lr': self.joint_opt_cam_lr},
                                                 {'params': [cam_poses_end], 'lr': self.joint_opt_cam_lr}])

        if idx == self.cfg['mapping']['keyframe_every']:
            lr_factor_not_first = self.cfg['mapping']['lr_factor']
            self.eslam.map_optimizer.param_groups[0]['lr'] = self.cfg['mapping']['lr']['decoders_lr'] * lr_factor_not_first
            self.eslam.map_optimizer.param_groups[1]['lr'] = self.cfg['mapping']['lr']['planes_lr'] * lr_factor_not_first
            self.eslam.map_optimizer.param_groups[2]['lr'] = self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor_not_first

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes, self.eslam.decoders)

            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                spline_poses = spline_linear_qt(cam_poses_start, cam_poses_end, NUM=self.cfg['blur']['spline_num'])
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(spline_poses)], dim=0)  ### qt->[4x4]
            else:
                if c2ws.shape[0]>1:
                    spline_poses = spline_linear_qt(matrix_to_cam_pose(c2ws_start[1:]), matrix_to_cam_pose(c2ws_end[1:]), NUM=self.cfg['blur']['spline_num'])
                    c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(spline_poses)], dim=0)
                else:
                    c2ws_ = cam_pose_to_matrix(spline_linear_qt(matrix_to_cam_pose(c2ws_start[:1]), matrix_to_cam_pose(c2ws_end[:1]), NUM=self.cfg['blur']['spline_num']))

            pixs_per_image = self.mapping_pixels // c2ws_.shape[0]

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples_blur(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device)

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                if c2ws.shape[0] > 1:
                    batch_rays_o_ = return_mid_exclude_first(batch_rays_o, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
                    batch_rays_d_ = return_mid_exclude_first(batch_rays_d, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
                else:
                    batch_rays_o_ = return_mid_continuous(batch_rays_o, spline_num=self.cfg['blur']['spline_num'])
                    batch_rays_d_ = return_mid_continuous(batch_rays_d, spline_num=self.cfg['blur']['spline_num'])
                det_rays_o = batch_rays_o_.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d_.clone().detach().unsqueeze(-1)
                t = (self.eslam.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth

                if c2ws.shape[0] > 1:
                    ### here maybe something wrong
                    # inside_mask_blur = torch.cat((inside_mask[:pixs_per_image],
                    #                          inside_mask[pixs_per_image:].repeat_interleave(self.cfg['blur']['spline_num'], dim=0)), dim=0)
                    inside_mask_blur = torch.cat((inside_mask[:pixs_per_image],
                                                  inside_mask[pixs_per_image:].reshape(kf_num, -1).repeat(1, self.cfg['blur']['spline_num']).flatten()), dim=0)
                else:
                    inside_mask_blur = inside_mask.repeat(self.cfg['blur']['spline_num'])

            batch_rays_d = batch_rays_d[inside_mask_blur]
            batch_rays_o = batch_rays_o[inside_mask_blur]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            ### render gt depth
            if c2ws.shape[0] > 1:

                batch_gt_depth_ = torch.cat((batch_gt_depth[:pixs_per_image], batch_gt_depth[pixs_per_image:].reshape(kf_num, -1).repeat(1, self.cfg['blur']['spline_num']).flatten()))
                # blur_batch_gt_depth_ = batch_gt_depth[pixs_per_image:].reshape(kf_num, -1).repeat(1, self.cfg['blur']['spline_num']).flatten()
                # blur_batch_gt_depth_repeat = batch_gt_depth[pixs_per_image:].repeat_interleave(self.cfg['blur']['spline_num'], dim=0)
            else:
                batch_gt_depth_ = batch_gt_depth.repeat(self.cfg['blur']['spline_num'])

            depth, color, sdf, z_vals = self.eslam.renderer.render_batch_ray(all_planes, self.eslam.decoders,batch_rays_d,
                                                                       batch_rays_o, device, self.eslam.truncation,
                                                                       gt_depth=batch_gt_depth_)
            depth_mask = (batch_gt_depth > 0)

            ## SDF losses
            if c2ws.shape[0] > 1:
                sdf = return_mid_exclude_first(sdf, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
                z_vals = return_mid_exclude_first(z_vals, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
            else:
                sdf = return_mid_continuous(sdf, spline_num=self.cfg['blur']['spline_num'])
                z_vals = return_mid_continuous(z_vals, spline_num=self.cfg['blur']['spline_num'])
            loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Color loss
            if c2ws.shape[0] > 1:
                color = return_blur_color_exclude_first(color, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
            else:
                color = return_blur_color(color, spline_num=self.cfg['blur']['spline_num'])
            loss = loss + self.w_color * torch.square(batch_gt_color - color).mean()

            ### Depth loss
            if c2ws.shape[0] > 1:
                depth = return_mid_exclude_first(depth, spline_num=self.cfg['blur']['spline_num'], batch=pixs_per_image)
            else:
                depth = return_mid_continuous(depth, spline_num=self.cfg['blur']['spline_num'])
            loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            self.eslam.map_optimizer.zero_grad()
            if self.joint_opt:
                camera_optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.eslam.map_optimizer.step()
            if self.joint_opt:
                camera_optimizer.step()

        if self.joint_opt:
            # put the updated camera poses back
            spline_poses = spline_linear_qt(cam_poses_start.detach(), cam_poses_end.detach(), NUM=self.cfg['blur']['spline_num'])
            cam_poses = return_mid(spline_poses, self.cfg['blur']['spline_num'])
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())
            optimized_c2ws_start = cam_pose_to_matrix(cam_poses_start.detach())
            optimized_c2ws_end = cam_pose_to_matrix(cam_poses_end.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    self.eslam.keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    self.eslam.keyframe_dict[frame]['est_c2w_start'] = optimized_c2ws_start[camera_tensor_id]
                    self.eslam.keyframe_dict[frame]['est_c2w_end'] = optimized_c2ws_end[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]
                    cur_c2w_start = optimized_c2ws_start[-1]
                    cur_c2w_end = optimized_c2ws_end[-1]

        return cur_c2w, cur_c2w_start, cur_c2w_end

    def first_frame_mapping(self, batch):
        idx: int = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        assert idx == 0

        if self.eslam.verbose:
            print(Fore.GREEN)
            print("First frame mapping...")
            print(Style.RESET_ALL)

        gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
        gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

        ## Fixing the first camera pose
        cur_c2w = gt_c2w
        self.eslam.estimate_c2ws[idx] = cur_c2w
        self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, cur_c2w.squeeze(),
                                  self.eslam.get_all_planes(), self.eslam.decoders)

        lr_factor = self.cfg['mapping']['lr_first_factor']
        iters = self.cfg['mapping']['iters_first']

        _ = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, cur_c2w)

        # add new frame to keyframe set
        self.eslam.keyframe_list.append(idx)
        self.eslam.keyframe_dict.append({
            'gt_c2w': gt_c2w,
            'idx': idx,
            'color': gt_color.to(self.keyframe_device),
            'depth': gt_depth.to(self.keyframe_device),
            'est_c2w': cur_c2w.clone()
        })

        # self.eslam.mapping_first_frame[0] = 1  # mapping of first frame is done, can begin tracking
        self.eslam.gt_c2ws[idx] = gt_c2w.clone()
        # self.eslam.pre_c2w = cur_c2w.clone()

    ### do not model blur
    def first_frame_mapping_blur(self, batch):
        idx: int = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        assert idx == 0

        if self.eslam.verbose:
            print(Fore.GREEN)
            print("First frame mapping...")
            print(Style.RESET_ALL)

        gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
        gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

        ## Fixing the first camera pose
        cur_c2w = gt_c2w
        self.eslam.estimate_c2ws[idx] = cur_c2w
        self.eslam.estimate_c2ws_start[idx] = cur_c2w
        self.eslam.estimate_c2ws_end[idx] = cur_c2w
        self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, cur_c2w.squeeze(),
                                  self.eslam.get_all_planes(), self.eslam.decoders)

        lr_factor = self.cfg['mapping']['lr_first_factor']
        iters = self.cfg['mapping']['iters_first']

        _ = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, cur_c2w)

        # add new frame to keyframe set
        self.eslam.keyframe_list.append(idx)
        self.eslam.keyframe_dict.append({
            'gt_c2w': gt_c2w,
            'idx': idx,
            'color': gt_color.to(self.keyframe_device),
            'depth': gt_depth.to(self.keyframe_device),
            'est_c2w': cur_c2w.clone(),
            'est_c2w_start': cur_c2w.clone(),
            'est_c2w_end': cur_c2w.clone()
        })

        # self.eslam.mapping_first_frame[0] = 1  # mapping of first frame is done, can begin tracking
        self.eslam.gt_c2ws[idx] = gt_c2w.clone()
        # self.eslam.pre_c2w = cur_c2w.clone()

    def current_frame_mapping(self, batch):
        pass

    def global_BA(self, batch):
        idx = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        if self.eslam.verbose:
            print(Fore.GREEN)
            print("Mapping Frame ", idx)
            print(Style.RESET_ALL)

        gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
        gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

        cur_c2w = self.eslam.estimate_c2ws[idx]

        lr_factor = self.cfg['mapping']['lr_factor']
        iters = self.cfg['mapping']['iters']

        ## Deciding if camera poses should be jointly optimized
        self.joint_opt = (len(self.eslam.keyframe_list) > 4) and self.cfg['mapping']['joint_opt']

        if self.cfg['tracking']['gt_camera']:
            cur_c2w = gt_c2w
        else:
            cur_c2w = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, cur_c2w)

        # self.visualizer.save_imgs(idx * 10000, 0, gt_depth, gt_color, cur_c2w.squeeze(),
        #                           self.eslam.get_all_planes(), self.eslam.decoders)

        if self.joint_opt:
            self.eslam.estimate_c2ws[idx] = cur_c2w

        # add new frame to keyframe set
        self.eslam.keyframe_list.append(idx)
        self.eslam.keyframe_dict.append({
            'gt_c2w': gt_c2w,
            'idx': idx,
            'color': gt_color.to(self.keyframe_device),
            'depth': gt_depth.to(self.keyframe_device),
            'est_c2w': cur_c2w.clone()
        })
        if self.eslam.verbose:
            print('Added keyframe: ', idx)


    def global_BA_blur(self, batch):
        idx = batch["frame_id"].item()
        gt_color = batch["rgb"]
        gt_depth = batch["depth"]
        gt_c2w = batch["c2w"]

        if self.eslam.verbose:
            print(Fore.GREEN)
            print("Mapping Frame ", idx)
            print(Style.RESET_ALL)

        gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
        gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
        gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

        cur_c2w = self.eslam.estimate_c2ws[idx]
        cur_c2w_start = self.eslam.estimate_c2ws_start[idx]
        cur_c2w_end = self.eslam.estimate_c2ws_end[idx]

        lr_factor = self.cfg['mapping']['lr_factor']
        iters = self.cfg['mapping']['iters']

        ## Deciding if camera poses should be jointly optimized
        self.joint_opt = (len(self.eslam.keyframe_list) > 4) and self.cfg['mapping']['joint_opt']

        cur_c2w, cur_c2w_start, cur_c2w_end = self.optimize_mapping_blur(iters, lr_factor, idx, gt_color, gt_depth, gt_c2w, cur_c2w, cur_c2w_start, cur_c2w_end)

        # self.visualizer.save_imgs(idx*10000, 0, gt_depth, gt_color, cur_c2w.squeeze(),
        #                           self.eslam.get_all_planes(), self.eslam.decoders)

        if self.joint_opt:
            self.eslam.estimate_c2ws[idx] = cur_c2w
            self.eslam.estimate_c2ws_start[idx] = cur_c2w_start
            self.eslam.estimate_c2ws_end[idx] = cur_c2w_end

        # add new frame to keyframe set
        self.eslam.keyframe_list.append(idx)
        self.eslam.keyframe_dict.append({
            'gt_c2w': gt_c2w,
            'idx': idx,
            'color': gt_color.to(self.keyframe_device),
            'depth': gt_depth.to(self.keyframe_device),
            'est_c2w': cur_c2w.clone(),
            'est_c2w_start': cur_c2w_start.clone(),
            'est_c2w_end': cur_c2w_end.clone()
        })
        if self.eslam.verbose:
            print('Added keyframe: ', idx)