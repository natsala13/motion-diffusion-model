import random

from tqdm import tqdm
import numpy as np
from yacs.config import CfgNode as CN
import torch
from torch.utils.data import DataLoader, Dataset

from data_loaders.interhuman.interhuman import InterHumanDataset, InterGenNormalizer

# from datasets.evaluator_models import InterCLIP



_C = CN(new_allowed=True)

def default_config() -> CN:
    """
    Get a yacs CfgNode object with the default config values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
def get_config(config_file: str, merge: bool = True) -> CN:
    """
    Read a config file and optionally merge it with the default config file.
    Args:
      config_file (str): Path to config file.
      merge (bool): Whether to merge with the default config or not.
    Returns:
      CfgNode: Config as a yacs CfgNode object.
    """
    if merge:
      cfg = default_config()
    else:
      cfg = CN(new_allowed=True)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


class EvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats, scale=1.0):
        self.normalizer = InterGenNormalizer()
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                _   , text, _, _, motion_lens, _ = data
                batch = {}
                if i in mm_idxs:
                    batch["text"] = list(text) * mm_num_repeats
                else:
                    batch["text"] = list(text)
                batch["motion_lens"] = motion_lens

                motions_output = self.model.forward_test(batch, scale=scale)
                motions_output = motions_output[:, :, :524, ...].reshape(motions_output.shape[0], motions_output.shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
                # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
                # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length


                sub_dict = {'motion1': motions_output[0, :,0],
                            'motion2': motions_output[0, :,1],
                            'motion_lens': motion_lens[0],
                            'text': text[0]}
                generated_motions.append(sub_dict)
                if i in mm_idxs:
                    mm_sub_dict = {'mm_motions': motions_output,
                                   'motion_lens': motion_lens[0],
                                    'text': text[0]}
                    mm_generated_motions.append(mm_sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens, -1


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_intergen_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, guidance_param=1.0):
    # Currently the configurations of two datasets are almost the same
    print(f'Generating motion using {model.__class__}')
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, scale=guidance_param)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader