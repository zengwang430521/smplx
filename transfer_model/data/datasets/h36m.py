# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

from typing import Optional, Tuple

import sys
import os
import os.path as osp

import numpy as np
# from psbody.mesh import Mesh
import trimesh

import torch
from torch.utils.data import Dataset
from loguru import logger


class H36M(Dataset):
    def __init__(self, npz_file='transfer_data/h36m_train_new.npz'):
        data = np.load(npz_file)
        self.poses = data['pose']
        self.betas = data['shape']

    def __len__(self) -> int:
        return len(self.poses)

    def __getitem__(self, index):
        return {
            'pose': self.poses[index].copy().astype(np.float32),
            'beta': self.betas[index].copy().astype(np.float32),
            'indices': index,
            # 'vertex_name': f'{index:06d}.npy',
            # 'para_name': f'{index:06d}.npz',
        }
