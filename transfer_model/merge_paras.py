import numpy as np
from scipy.spatial.transform import Rotation as R
import os.path as osp
from tqdm import tqdm

source_file = 'transfer_data/h36m_train_new.npz'
target_file = 'transfer_data/h36m_train_smplx.npz'
para_folder = 'transfer_data/h36m_paras'

source_data = np.load(source_file)

target_data = {key: source_data[key] for key in source_data.keys()}
num_item = len(target_data['imgname'])

pose_x, beta_x = [], []

for index in tqdm(range(num_item)):
    p_name = osp.join(para_folder, f'{index:06d}.npz')
    p_data = np.load(p_name)
    beta = p_data['betas']
    beta_x.append(beta)

    full_rotmat = p_data['full_pose']
    full_pose = []
    for rotmat in full_rotmat:
        rot = R.from_matrix(rotmat)
        vec = rot.as_rotvec()
        full_pose.append(vec)
    full_pose = np.stack(full_pose)
    pose_x.append(full_pose)

pose_x = np.stack(pose_x)
beta_x = np.stack(beta_x)
target_data['poses_smplx'] = pose_x
target_data['betas_smplx'] = beta_x

np.savez(target_file, **target_data)
