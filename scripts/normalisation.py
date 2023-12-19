import os
from os.path import join as pjoin
from argparse import ArgumentParser

import torch
import tqdm
import numpy as np

from data_loaders.humanml.common.quaternion import qmul_np, qinv_np, qrot_np
from utils.intergen_utils import load_motion, process_motion_np, rigid_transform
from data_loaders.interhuman.interhuman import InterHumanDataset, InterGenNormalizer
from data_loaders.humanml.data.dataset import HumanML3D


MIN_LENGTH = 15
DATAPATH = '../InterGen/data/interhuman/'


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument("--num_frames", default=-1, type=int, help='number of frames to load')
    parser.add_argument("--plot", help="plot pre saved calculus.", action="store_true")
    parser.add_argument("--number", default=-1, type=int, help="number of motion to load.")
    return parser.parse_args()


def get_motion_files(num_frames: int=-1):
    data_list = open(pjoin(DATAPATH, "split/test.txt"), "r").readlines()
    data_list = [int(file[:-1]) for file in data_list]


    files = [file for file in os.listdir(pjoin(DATAPATH, 'motions_processed/person1')) 
             if int(file.split('.')[0]) in data_list]
    num_frames = num_frames if num_frames > 0 else len(files)
    
    return files[:num_frames]

def translate_rotate_motion(motion, root_quat_init1, root_quat_init2, root_pos_init1, root_pos_init2):
    r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
    angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

    xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
    relative = np.concatenate([angle, xz], axis=-1)[0]
    motion = rigid_transform(relative, motion)

    return motion


def load_all_motions(num_frames: int=-1):
    files = get_motion_files(num_frames)
    motions = []

    for file in tqdm.tqdm(files):
        file_path_person1 = pjoin(DATAPATH, 'motions_processed/person1', file)
        file_path_person2 = file_path_person1.replace("person1", "person2")

        motion1, _ = load_motion(file_path_person1, MIN_LENGTH, swap=False)
        if motion1 is None:
            continue
        
        motion2, _ = load_motion(file_path_person2, MIN_LENGTH, swap=False)

        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        motion2 = translate_rotate_motion(motion2, root_quat_init1, root_quat_init2, root_pos_init1, root_pos_init2)

        motions += [np.concatenate((motion1, motion2), axis=1)]

    # return motions
    return np.concatenate(motions, axis=0)





if __name__ == '__main__':
    args = prepare_args()

    # data_cfg = get_config("configs/datasets.yaml").interhuman_test
    # data_cfg.SIZE = args.num_frames

    normalize = InterGenNormalizer()
    intergen = InterHumanDataset(split='test', num_frames=args.num_frames)
    humanml = HumanML3D(split='test', num_frames=args.num_frames, mode='gt')
    motions = load_all_motions(args.num_frames)

    mean = np.mean(motions, axis=0)
    std = np.std(motions, axis=0)

    intergen_motion0 = intergen[0][2]
    humanml_motion0 = humanml[0][4]

    import ipdb;ipdb.set_trace()



    