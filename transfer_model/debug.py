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

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from smplx import build_layer

from data import build_dataloader
from transfer_model import run_fitting
from utils import read_deformation_transfer, np_mesh_to_o3d
from config import parse_args
import trimesh

# from fvcore.common.config import CfgNode as CN
# _C = CN
#
# _C.datasets = CN
# _C.datasets.mesh_folder = CN
# _C.datasets.mesh_folder.data_folder = 'transfer_data/meshes/smpl'
# _C.deformation_transfer_path = 'transfer_data/smpl2smplx_deftrafo_setup.pkl'
# _C.mask_ids_fname = 'smplx_mask_ids.npy'
# _C.summary_steps = 100
# _C.edge_fitting = CN
# _C.edge_fitting.per_part = False
# _C.optim = CN
# _C.optim.type = 'trust-ncg'
# _C.optim.maxiters = 100
# _C.optim.gtol = 1e-06
# _C.body_model = CN
# _C.body_model.model_type = "smplx"
# _C.body_model.gender = "neutral"
# _C.body_model.folder = "transfer_data/body_models"
# _C.body_model.use_compressed = False
# _C.body_model.use_face_contour = True
# _C.body_model.smplx = CN
# _C.body_model.smplx.betas = CN
# _C.body_model.smplx.betas.num = 10
# _C.body_model.smplx.expression = CN
# _C.body_model.smplx.expression.num = 10

from smplx.lbs import batch_rodrigues
from data.datasets.h36m import H36M
import torch.utils.data as dutils

def main() -> None:

    exp_cfg = parse_args()
    # exp_cfg = _C

    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    # logger.add(
    #     lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
    #     colorize=True)

    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    mesh_path = 'transfer_data/meshes/smpl/01.ply'
    sample_mesh = trimesh.load(mesh_path, process=False)
    sample_face = np.asarray(sample_mesh.faces, dtype=np.int32)
    sample_face = torch.tensor(sample_face).unsqueeze(0).to(device)

    smpl = build_layer(model_path, model_type='smpl').to(device)

    h36m_dataset = H36M()

    batch_size = exp_cfg.batch_size
    num_workers = exp_cfg.datasets.num_workers

    logger.info(
        f'Creating dataloader with B={batch_size}, workers={num_workers}')
    dataloader = dutils.DataLoader(h36m_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False)
    vertex_dir = 'transfer_data/h36m_vertices'
    para_dir = 'transfer_data/h36m_paras'
    if not osp.exists(vertex_dir):
        os.makedirs(vertex_dir)
    if not osp.exists(para_dir):
        os.makedirs(para_dir)

    for ii, batch in enumerate(tqdm(dataloader)):
        print(f'{ii:06d}/{len(dataloader):06d}')
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)

        indices = batch['indices']
        batch_size = len(indices)
        valid = torch.ones(batch_size, dtype=torch.bool, device=device)
        for i_item in range(batch_size):
            index = indices[i_item]
            v_name = f'{index:06d}.npy'
            p_name = f'{index:06d}.npz'
            if osp.exists(osp.join(vertex_dir, v_name)) \
                    and osp.exists(osp.join(para_dir, p_name)):
                valid[i_item] = 0

        for key in batch:
            batch[key] = batch[key][valid]

        # load again
        indices = batch['indices']
        pose, beta = batch['pose'], batch['beta']
        batch_size = pose.shape[0]
        if batch_size == 0:
            continue

        pose = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
        body_pose = pose[:, 1:]
        global_orient = pose[:, 0]

        mesh = smpl(betas=beta,
                    body_pose=body_pose,
                    global_orient=global_orient)
        vertices = mesh.vertices
        batch['vertices'] = vertices
        batch['faces'] = sample_face.expand([batch_size, -1, -1])
        batch['indices'] = torch.zeros(1).long().to(device)
        var_dict = run_fitting(exp_cfg, batch, body_model, def_matrix, mask_ids)

        for ii in range(batch_size):
            index = indices[ii]
            v_name = f'{index:06d}.npy'
            p_name = f'{index:06d}.npz'

            v_smplx = var_dict['vertices'][ii].detach().cpu().numpy()
            np.save(osp.join(vertex_dir, v_name), v_smplx)

            para_dict = {}
            for key in var_dict.keys():
                if key not in ['faces', 'vertices']:
                    para_dict[key] = var_dict[key][ii].detach().cpu().numpy()
            np.savez(osp.join(para_dir, p_name), **para_dict)

        del var_dict, batch, mesh, pose, beta, body_pose, global_orient

        # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
