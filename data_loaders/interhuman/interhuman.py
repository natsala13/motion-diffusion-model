import os
import random
from os.path import join as pjoin

import torch
from torch import Tensor
from torch.utils import data 
import einops
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from data_loaders.humanml.common.quaternion import qmul_np, qinv_np, qrot_np
from utils.intergen_utils import load_motion, process_motion_np, rigid_transform
from data_loaders.tensors import collate
 

NAMES = ['Hips', 'RightUpLeg', 'LeftUpLeg', 'Spine', 'RightLeg', 'LeftLeg', 'Spine1',
        'RightFoot', 'LeftFoot', 'Spine2', 'RightToe', 'LeftToe', 'Neck',
        'RightShoulder', 'LeftShoulder', 'Head', 'RightArm', 'LeftArm', 
        'RightForeArm', 'LeftForeArm', 'LeftHand', 'RightHand' ]

MINIMAL_NAMES = ['Hips', 'Head', 'LeftHand', 'RightHand', 'LeftLeg', 'RightLeg']


class InterHumanDataset(data.Dataset):
    def __init__(self, split, datapath='../InterGen/data/interhuman/',
                  num_frames=-1, normalize=True, **kwargs):
        self.dataname = 'intergen'
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        self.normalize = normalize
        self.normalizer = InterGenNormalizer()

        # self.motion_rep = global
        self.data_list = []
        self.motion_dict = []

        self.cache = True

        data_list, ignore_list = self._get_data_list(datapath, split)

        index = 0
        root, dirs, files = next(os.walk(pjoin(datapath, 'motions_processed/person1')))
        
        files = [file for file in files if int(file.split('.')[0]) in data_list]
        num_frames = num_frames if num_frames > 0 else len(files)

        for file in tqdm(files[:num_frames]):
            motion_name = file.split(".")[0]
            if file.split(".")[0]+"\n" in ignore_list: # or int(motion_name)>1000
                print("ignore: ", file)
                continue

            file_path_person1 = pjoin(root, file)
            file_path_person2 = pjoin(root.replace("person1", "person2"), file)
            text_path = file_path_person1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")

            texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
            texts_swap = [item.replace("\n", "").replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                            .replace("clockwise", "tmp").replace("counterclockwise","clockwise").replace("tmp","counterclockwise") for item in texts]

            if self.cache:
                motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)
                motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)
                if motion1 is None:
                    continue
                for motion1, motion2 in [[motion1, motion2], [motion2, motion1], [motion1_swap, motion2_swap], [motion2_swap, motion1_swap]]:
                    motion1, motion2, distance_matrix = self._process_motion(motion1, motion2)
                    self.motion_dict.append([motion1, motion2, distance_matrix])

            else:
                self.motion_dict.append([file_path_person1, file_path_person2])
                self.motion_dict.append([file_path_person1, file_path_person2])
            # if self.cache:
            #     motion1, motion1_swap = load_motion(file_path_person1, self.min_length, swap=True)
            #     motion2, motion2_swap = load_motion(file_path_person2, self.min_length, swap=True)
            #     if motion1 is None:
            #         continue
            #     self.motion_dict.append([motion1, motion2])
            #     self.motion_dict.append([motion1_swap, motion2_swap])
            # else:
            #     self.motion_dict.append([file_path_person1, file_path_person2])
            #     self.motion_dict.append([file_path_person1, file_path_person2])

            self.data_list.append({"name": motion_name, "motion_id": index, "swap":False, "texts":texts})
            # TODO: what about all the reversed motions from above?
            if split == "train":
                self.data_list.append({"name": motion_name+"_rotate", "motion_id": index+1, "swap": False, "texts": texts})
                self.data_list.append({"name": motion_name+"_swap", "motion_id": index+2, "swap": True, "texts": texts_swap})
                self.data_list.append({"name": motion_name+"_rotate_swap", "motion_id": index+3, "swap": True, "texts": texts_swap})
                

            index += 4

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)
    
    @staticmethod
    def _get_data_list(datapath, split):
        ignore_list = []
        try:
            ignore_list = open(os.path.join(datapath, "split/ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        
        data_list = []
        if split == "train":
            try:
                data_list = open(os.path.join(datapath, "split/train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif split == "val":
            try:
                data_list = open(os.path.join(datapath, "split/val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif split == "test":
            try:
                data_list = open(os.path.join(datapath, "split/test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        random.shuffle(data_list)
        data_list = [int(file[:-1]) for file in data_list]

        return data_list, ignore_list
    
    def _process_motion(self, motion1, motion2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
        motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

        xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        motion2 = rigid_transform(relative, motion2)
    
        if self.normalize:
            motion1, motion2 = map(self.normalizer.forward, [motion1, motion2])
        
        distance_matrix = self.distance_matrix(motion1, motion2)

        return motion1, motion2, distance_matrix

    def __len__(self):
        return self.real_len()

    def __getitem__(self, item):
        idx = item % self.real_len()
        data_list = self.data_list[idx]

        name = data_list["name"]
        motion_id = data_list["motion_id"]
        swap = data_list["swap"]
        text = random.choice(data_list["texts"]).strip()

        if self.cache:
            full_motion1, full_motion2, distance_matrix = self.motion_dict[motion_id]
        else:
            file_path1, file_path2 = self.motion_dict[motion_id]
            motion1, motion1_swap = load_motion(file_path1, self.min_length, swap=swap)
            motion2, motion2_swap = load_motion(file_path2, self.min_length, swap=swap)
            if swap:
                full_motion1 = motion1_swap
                full_motion2 = motion2_swap
            else:
                full_motion1 = motion1
                full_motion2 = motion2
            if full_motion1 is None or full_motion2 is None:
                return self.__getitem__(item+1)

            motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion1, 0.001, 0, n_joints=22)
            motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion2, 0.001, 0, n_joints=22)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]
            motion2 = rigid_transform(relative, motion2)

            if self.normalize:
                motion1, motion2 = map(self.normalizer.forward, [motion1, motion2])

            distance_matrix = self.distance_matrix(motion1, motion2)

        length = full_motion1.shape[0]
        if length > self.max_length:
            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            motion1 = full_motion1[idx:idx + gt_length]
            motion2 = full_motion2[idx:idx + gt_length]
            distance_matrix = distance_matrix[idx:idx + gt_length]

        else:
            gt_length = min(length, self.max_gt_length )
            motion1 = full_motion1[:gt_length]
            motion2 = full_motion2[:gt_length]
            distance_matrix = distance_matrix[:gt_length]

        # if np.random.rand() > 0.5:
        #     motion1, motion2 = motion2, motion1
        #     distance_matrix = self._transpsoe_distance_matrix(distance_matrix)
            
        gt_length = len(motion1)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            batch_size = motion1.shape[1]
            padding_zeros = np.zeros((padding_len, batch_size))
            motion1 = np.concatenate((motion1, padding_zeros), axis=0)
            motion2 = np.concatenate((motion2, padding_zeros), axis=0)
            distance_matrix = np.concatenate((distance_matrix,
                                               np.zeros((padding_len,
                                                          distance_matrix.shape[1]))), axis=0)

        assert len(motion1) == self.max_gt_length
        assert len(motion2) == self.max_gt_length
        assert len(distance_matrix) == self.max_gt_length

        return name, text, motion1, motion2, gt_length, distance_matrix

    @staticmethod
    def distance_matrix(motion1: np.ndarray, motion2: np.ndarray) -> np.ndarray:
        '''calculate distance between min joints of motion1 to min joints of motion2.'''
        order = [NAMES.index(edge) for edge in MINIMAL_NAMES]
        num_joints = len(NAMES)
        motion1 = motion1[:, :num_joints * 3].reshape(-1, num_joints, 3)[:, order]
        motion2 = motion2[:, :num_joints * 3].reshape(-1, num_joints, 3)[:, order]

        num_joints = len(MINIMAL_NAMES)
        motion1_extend = einops.repeat(motion1, 'b n f -> b (c n) f', c=num_joints)
        motion2_extend = einops.repeat(motion2, 'b n f -> b (n c) f', c=num_joints)

        distance = motion1_extend - motion2_extend
        distance_matrix = einops.rearrange(distance, 'b f dim -> b (f dim)')
        # distance_matrix = norm(distance_matrix, axis=-1)

        return distance_matrix

    @staticmethod
    def _transpsoe_distance_matrix(matrix: np.ndarray) -> np.ndarray:
        num_joints = int(np.sqrt(matrix.shape[-1]))
        return np.transpose(matrix.reshape(-1, num_joints, num_joints), axes=[0,2,1]).reshape(-1, num_joints**2)

    def shape(self):
        pass

    @property
    def opt(self):
        class Opt:
            def __init__(self, max_length) -> None:
                self.max_motion_length = max_length

        return Opt(self.max_length)


# an adapter to our collate func
def interhuman_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[2].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[1],
        'lengths': b[4],
    } for b in batch]
    return collate(adapted_batch)


def interhuman_couple_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(np.concatenate((b[2], b[3]), axis=1).T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[1],
        'lengths': b[4],
        'tokens': b[1]  # For compatability reasons
    } for b in batch]
    return collate(adapted_batch)


def interhuman_time_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(np.concatenate((b[2], b[3]), axis=0).T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[1],
        'lengths': b[4],
        'tokens': b[1]
    } for b in batch]
    return collate(adapted_batch)


def interaction_matrix_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(np.concatenate((b[2], b[3], b[5]), axis=1).T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[1],
        'lengths': b[4],
        'tokens': b[1]  # For compatability reasons
    } for b in batch]
    return collate(adapted_batch)


class Base_Normalizer():
    def __init__(self, mean: Tensor, std: Tensor) -> None:
        self.motion_mean = mean
        self.motion_std = std

    def forward(self, x):
        x = (x - self.motion_mean) / self.motion_std
        return x

    def backward(self, x):
        x = x * self.motion_std + self.motion_mean
        return x


class InterGenNormalizer(Base_Normalizer):
    def __init__(self):
        mean = np.load("../InterGen/data/global_mean.npy")
        std = np.load("../InterGen/data/global_std.npy")

        super().__init__(mean, std)

class HumanMlNormalizer(Base_Normalizer):
    def __init__(self):
        mean = np.load('dataset/HumanML3D/Mean.npy')
        std = np.load('dataset/HumanML3D/Std.npy')
        super().__init__(mean, std)
