import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from data_loaders.tensors import collate
# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.rot2xyz = self.model.rot2xyz
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = 'text'

    def forward(self, x, timesteps, y=None):
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1, 1) * (out - out_uncond))
    
    def forward_test(self, batch, scale=1.0):
        assert self.model.diffusion != None

        batch_size = len(batch['text'])
        n_frames = batch['motion_lens'][0]

        # batch_size, n_joints, n_features, n_frames = x.shape[0]
        collate_args = [{'inp': torch.zeros(n_frames), 'text': batch['text'][0], 'tokens': None, 'lengths': n_frames}] * batch_size
        _, model_kwargs = collate(collate_args)
        device = next(self.parameters()).device
        model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}
        if scale != 1.:
            model_kwargs['y']['scale'] = torch.ones(batch_size, device='cuda') * scale

        sample = self.model.diffusion.p_sample_loop(
            self,
            # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
            (batch_size, self.njoints, self.nfeats, n_frames),  # BUG FIX
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        return {'output': sample.permute(0, 3, 1, 2)}

