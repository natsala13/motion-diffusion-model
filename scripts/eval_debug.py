''' I want to load all files from debug/embeddings where each of the file contain a batch of 
    98 motions of 2 persons, along with their text and motion embeddings.
    After that I want to calculate their mean and std and calculate their FID score 
    according to groundtruth as done in the file eval/eval_intergen.py'''

import os
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.intergen_eval_utils import get_config
from data_loaders.interhuman.interhuman import InterHumanDataset
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorIntergenWrapper
from eval.eval_intergen import evaluate_matching_score, evaluate_fid


NULL_FILE = open('/dev/null', 'w')


def load_all_embeddings(root_path: str):
    files = os.listdir(root_path)
    all_embeddingss = []

    for file in tqdm(files):
        embeddings = np.load(pjoin(root_path, file))
        all_embeddingss.append(embeddings['motion_embeddings'])


    return np.concatenate(all_embeddingss, axis=0)


def evaluate_tools():
    gt_dataset = InterHumanDataset(split='test', normalize=False)
    eval_gt_data = DataLoader(gt_dataset, batch_size=16,
                                           shuffle=False, num_workers=0, drop_last=True)
    eval_config = get_config('../InterGen/configs/eval_model.yaml')
    eval_wrapper = EvaluatorIntergenWrapper(eval_config, 'cuda')

    return eval_gt_data, eval_wrapper


def load_ground_truth_embeddings(eval_gt_data, eval_wrapper):
    _, _, acti_dict = evaluate_matching_score(eval_wrapper, {'gt': eval_gt_data}, NULL_FILE)
    
    return acti_dict['gt']


if __name__ == '__main__':
    gt_loader, eval_wrapper = evaluate_tools()

    gen_embeddings = load_all_embeddings('debug/embeddings')
    inter_embeddings = load_all_embeddings('../InterGen/debug/embeddings2')
    # gt_embeddings = load_ground_truth_embeddings(gt_loader, eval_wrapper)
    embeddings = {'gen': gen_embeddings, 'inter': inter_embeddings}


    fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, embeddings, NULL_FILE)
