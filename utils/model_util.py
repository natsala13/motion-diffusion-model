import re

import torch
import numpy as np

from model.mdm import MDM
from model.mdm_any import MdmAny, MdmAttend, MdmTime, MdmSimetric
from model.tedi import Unet
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    # state_dict = {re.sub('^model.(model.)?', '', key): value for key, value in state_dict['state_dict'].items()}
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    diffusion = create_gaussian_diffusion(args)
    if args.arch == 'mdm_any':
        model = MdmAny(input_features=262, diffusion=diffusion)
    elif args.arch == 'mdm_attend':
        model = MdmAttend(input_features=263, diffusion=diffusion)
    elif args.arch == 'mdm_time':
        model = MdmTime(input_features=262, diffusion=diffusion)
    elif args.arch == 'tedi':
        model = Unet(input_features=262, diffusion=diffusion)
    elif args.arch == 'mdm_symetric':
        model = MdmSimetric(input_features=262, diffusion=diffusion,
                            second_attention=args.second_attention,
                            zero_in_projection=args.zero_in_initiaization,
                            zero_out_projection=args.zero_out_initiaization)
    else:
        model = MDM(**get_model_args(args, data), diffusion=diffusion)

    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if data and hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1
    elif args.dataset == 'interhuman_solo':
        data_rep = 'interhuman_solo'
        njoints = 262
        nfeats = 1
    elif args.dataset == 'interhuman_matrix':
        data_rep = 'interhuman'
        njoints = 262 * 2 + 25 # 108
        nfeats = 1
    elif 'interhuman' in args.dataset:
        data_rep = 'interhuman'
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': args.ff_size, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'window_size': args.window_size}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, verbose=False):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch+1) * 1.0 / self.warmup
        return lr_factor
    