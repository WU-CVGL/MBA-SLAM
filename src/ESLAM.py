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
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/NICE_SLAM.py
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
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.tools.cull_mesh import cull_mesh
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

# spline
from src.utils.Spline import get_pose_interval

if TYPE_CHECKING:
    import argparse


class ESLAM(object):
    """
    ESLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking processes.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """
    def __init__(self, cfg: dict, args: argparse.Namespace):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        # last previous pose
        self.pre_c2w = None

        self.decoders = config.get_model(cfg).to(self.device)

        self.scale = cfg['scale']

        self.load_bound(cfg)
        self.init_planes(cfg)

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, num_workers=1, pin_memory=True, prefetch_factor=2
        )

        self.estimate_c2ws = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2ws_start = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2ws_end = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.gt_c2ws = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2ws_rel = torch.zeros((self.n_img, 4, 4), device=self.device)

        self.keyframe_dict = []
        self.keyframe_list = []

        # exposure time interval
        self.pose_interval = get_pose_interval(self.cfg['blur']['spline_num'], self.device)

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config,
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_dividable = cfg['planes_res']['bound_dividable']
        # enlarge the bound a bit to allow it dividable by bound_dividable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]
        self.decoders.bound = self.bound

    def get_all_planes(self):
        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz,
            self.c_planes_xy, self.c_planes_xz, self.c_planes_yz
        )
        return all_planes

    def get_all_planes_detached(self):
        all_planes = self.get_all_planes()
        return tuple([[plane.detach() for plane in planes] for planes in all_planes])

    def init_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        ####### Initializing Planes ############
        self.planes_xy, self.planes_xz, self.planes_yz = [], [], []
        self.c_planes_xy, self.c_planes_xz, self.c_planes_yz = [], [], []
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            self.planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01).to(self.device))
            self.planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01).to(self.device))
            self.planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01).to(self.device))

        for grid_res in c_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            self.c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01).to(self.device))
            self.c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01).to(self.device))
            self.c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01).to(self.device))

        for i in range(2):
            self.planes_xy[i] = nn.Parameter(self.planes_xy[i])
            self.planes_xz[i] = nn.Parameter(self.planes_xz[i])
            self.planes_yz[i] = nn.Parameter(self.planes_yz[i])
            self.c_planes_xy[i] = nn.Parameter(self.c_planes_xy[i])
            self.c_planes_xz[i] = nn.Parameter(self.c_planes_xz[i])
            self.c_planes_yz[i] = nn.Parameter(self.c_planes_yz[i])
        print('change planes to parameters')

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA
        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())

        planes_para = []
        planes_para.append(self.planes_xy[0])
        planes_para.append(self.planes_xy[1])
        planes_para.append(self.planes_xz[0])
        planes_para.append(self.planes_xz[1])
        planes_para.append(self.planes_yz[0])
        planes_para.append(self.planes_yz[1])

        c_planes_para = []
        c_planes_para.append(self.c_planes_xy[0])
        c_planes_para.append(self.c_planes_xy[1])
        c_planes_para.append(self.c_planes_xz[0])
        c_planes_para.append(self.c_planes_xz[1])
        c_planes_para.append(self.c_planes_yz[0])
        c_planes_para.append(self.c_planes_yz[1])

        # planes_para = []
        # for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
        #     for i, plane in enumerate(planes):
        #         planes_para.append(plane)  ### here if parameters()?
        #         # planes[i] = plane
        #
        # c_planes_para = []
        # for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
        #     for i, c_plane in enumerate(c_planes):
        #         c_planes_para.append(c_plane)
        #         # c_planes[i] = c_plane

        self.map_optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                      {'params': planes_para, 'lr': 0},
                                      {'params': c_planes_para, 'lr': 0}])

        lr_factor = self.cfg['mapping']['lr_first_factor']
        self.map_optimizer.param_groups[0]['lr'] = self.cfg['mapping']['lr']['decoders_lr'] * lr_factor
        self.map_optimizer.param_groups[1]['lr'] = self.cfg['mapping']['lr']['planes_lr'] * lr_factor
        self.map_optimizer.param_groups[2]['lr'] = self.cfg['mapping']['lr']['c_planes_lr'] * lr_factor

    def convert_relative_pose(self, frame_id):
        poses = torch.zeros((self.n_img, 4, 4), device=self.device)
        for i in range(frame_id):
            if i % self.keyframe_every == 0:
                poses[i] = self.estimate_c2ws[i]
            else:
                kf_id = i // self.keyframe_every
                kf_frame_id = kf_id * self.keyframe_every
                c2w_key = self.estimate_c2ws[kf_frame_id]
                delta = self.estimate_c2ws_rel[i]
                poses[i] = delta @ c2w_key

        return poses

    def run(self):
        """
        Begin.
        """
        self.create_optimizer()
        progress_bar = tqdm(enumerate(self.frame_loader), smoothing=0.05)
        for i, batch in progress_bar:
            # First frame mapping
            if i == 0:
                if not self.cfg['blur']['enable']:
                    self.mapper.first_frame_mapping(batch)
                else:
                    self.mapper.first_frame_mapping_blur(batch)
                # Tracking + Mapping
            else:
                if not self.cfg['blur']['enable']:
                    self.tracker.tracking_render(batch)
                    if i % self.keyframe_every == 0:
                        self.mapper.global_BA(batch)
                else:
                    self.tracker.tracking_render_blur_gtcamera(batch)
                    if i % self.keyframe_every == 0:
                        self.mapper.global_BA_blur(batch)

            # Save checkpoint
            if ((not (i == 0 and self.no_log_on_first_frame))and i % self.ckpt_freq == 0) or i == self.n_img - 1:
                self.logger.log(i, self.keyframe_list, self.convert_relative_pose(i))

            # Visualization
            if (i % self.mesh_freq == 0) and (not (i == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f'{self.output}/mesh/{i:05d}_mesh.ply'
                self.mesher.get_mesh(mesh_out_file, self.all_planes, self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                          estimate_c2ws=self.estimate_c2ws[:i + 1])
            if i == self.n_img - 1:
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                self.decoders.eval()
                self.mesher.get_mesh(mesh_out_file, self.get_all_planes_detached(), self.decoders, self.keyframe_dict, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2ws=self.estimate_c2ws)

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
