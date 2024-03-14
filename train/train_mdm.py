# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import os
import json

import wandb

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import NoPlatform, WandBPlatform, WandBSweepPlatform  # required for the eval operation


def create_save_dir(args):
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

def change_args_with_sweep(args, sweep_config: dict, name: str):
    print('##### ARGS #####')
    print(args)
    print('##### SWEEP CONFIG #####')
    print(sweep_config)
    
    for key, value in sweep_config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f'Invalid key {key} in sweep_config')
    
    if name is not None:
        setattr(args, 'save_dir', f'save/{name}')
    # setattr(args, 'num_frames', 100)  # TODO: Remove.

    print('##### NEW ARGS #####')
    print(args)
    print('###################')
    return args

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir,
                                         experiment_name=args.experiment_name,
                                         lr=args.lr, batch_size=args.batch_size,
                                         architecture=args.arch, latent=args.latent_dim)

    args = change_args_with_sweep(args, train_platform.config, train_platform.name)
    train_platform.report_args(args, name='Args')
    create_save_dir(args)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
