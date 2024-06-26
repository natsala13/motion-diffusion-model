import os
import re
import sys
import copy
import time
import signal
import functools
from functools import partial
from os.path import join as pjoin
from types import SimpleNamespace

import numpy as np
import blobfile as bf
from tqdm import tqdm
import torch
from torch.optim import AdamW
import einops

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc, eval_intergen
from data_loaders.get_data import get_dataset_loader
from utils.model_util import CosineWarmupScheduler
import data_loaders.humanml.utils.paramUtil as paramUtil
from sample.generate import generate_motion, denormalize_motion
from data_loaders.interhuman.interhuman import InterGenNormalizer
from data_loaders.humanml.utils.plot_script import plot_3d_motion_interaction

# Intergen evaluation imports.
from torch.utils.data import DataLoader
from data_loaders.interhuman.interhuman import InterHumanDataset
from utils.intergen_eval_utils import get_config, get_intergen_loader
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorIntergenWrapper

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
DEFAULT_GAMMA = 0.99998


class TrainLoop:
    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        # self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self._resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self.save_dir = args.save_dir
        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step > 0:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        match args.lr_method:
            case 'exp':
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=DEFAULT_GAMMA)
            case 'cos':
                self.lr_scheduler = CosineWarmupScheduler(optimizer=self.opt, warmup=None, max_iters=args.num_steps)
            case _:
                self.lr_scheduler = None

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper   , self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=args.num_frames,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=args.num_frames,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        elif 'interhuman' in args.dataset and args.eval_during_training:
            # self.eval_gt_data, gt_dataset = get_dataset_motion_loader(split='test')
            gt_dataset = InterHumanDataset(split='test', normalize=False, num_frames=args.num_frames)
            self.eval_gt_data = DataLoader(gt_dataset, batch_size=args.eval_batch_size,
                                           shuffle=True, num_workers=0, drop_last=True)

            eval_config = get_config('../InterGen/configs/eval_model.yaml')
            self.eval_wrapper = EvaluatorIntergenWrapper(eval_config, self.device)

            self.eval_data = {'test': partial(get_intergen_loader,
                                                args.eval_batch_size,  # has to be 96!
                                                model,
                                                gt_dataset,
                                                self.device,
                                                mm_num_samples=0,
                                                mm_num_repeats=0
                                                )
                                                }
        
        self.normalizer = InterGenNormalizer()  # For generation


        self.use_ddp = False
        signal.signal(signal.SIGUSR1, self._listen_to_signal_and_save)

    @property
    def resume_step(self) -> int:
        return self._resume_step + self.step

    def _listen_to_signal_and_save(self, signal_number, frame):
        '''save the model before task is killed...'''
        if signal_number == signal.SIGUSR1:
            print(f'SIGUSR1 recieved after {self.resume_step}, saving model gentely...')
            self.save()
            print('model saved')
            self.train_platform.close_preempting()
            print('platform closed, existing...')
            sys.exit(16)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.save_dir) or self.resume_checkpoint

        if resume_checkpoint:
            self._resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            self.model.load_state_dict(state_dict, strict=False)  # Due to Clip's weight that are not saved in checkpoint.

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.save_dir) or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self._resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):

        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data, position=0, leave=True):
                if not (not self.lr_anneal_steps or self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)  # (batch, ch , ? , time), humanml: (64, 263, 1, 196)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
                
                self.run_step(motion, cond)
                
                if self.step % self.log_interval == 0:  # log_interval: 1000, size(data): 10k, data/batch_size = ~150, --> log every +-6 epochs
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.resume_step, group_name='Loss')

                if self.resume_step > 0 and self.resume_step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.generate()
                    self.model.train()

                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1

            if not (not self.lr_anneal_steps or self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()

        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_interhumam_{(self.resume_step):09d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            if self.dataset == 'humanml':
                eval_dict = eval_humanml.evaluation(
                    self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                    replication_times=1, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            elif 'interhuman' in self.dataset:
                eval_dict = eval_intergen.evaluation(
                    self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                    replication_times=1, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            else:
                return

            metrices_to_report = ['FID_test', 'Diversity_test']  # 'MM Distance_test'
            for k, v in {key: eval_dict[key] for key in metrices_to_report}.items():
                self.train_platform.report_scalar(name=k, value=v,
                                                   iteration=self.resume_step,
                                                     group_name='Eval')
            # for k, v in eval_dict.items():
            #     if k.startswith('R_precision'):
            #         for i in range(len(v)):
            #             self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
            #                                               iteration=self.step + self.resume_step,
            #                                               group_name='Eval')
            #     else:
            #         self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
            #                                           group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def generate(self):
        n_frames = 210
        motion, cond = next(iter(self.data))
        cond['y']['text'] = cond['y']['text'][:1]
        cond['y']['length'] = torch.tensor([n_frames], device=self.device)
        cond['y']['mask'] = cond['y']['mask'][:1, ..., :n_frames]

        # import ipdb;ipdb.set_trace()
        sample = generate_motion(self.model, self.diffusion, cond, n_frames, guidance_param=1, batch_size=1, device=self.device)
        sample, _, _ = denormalize_motion(sample, self.normalizer, 1, n_frames, 'interhuman')

        sample = self.model.rot2xyz(x=sample, mask=None, pose_rep='xyz', glob=True, translation=True, jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None, get_rotations_back=False)
        
        motion = einops.rearrange(sample[0], 'p (J xyz) t -> p t J xyz', xyz=3)
        skeleton = paramUtil.t2m_kinematic_chain
        
        artifact_name = f'motion_{self.resume_step:09d}'
        artifact_path = f'{self.save_dir}/{artifact_name}.mp4'
        caption = cond['y']['text'][0]
        plot_3d_motion_interaction(artifact_path, skeleton, motion.numpy(), title=caption[0], fps=20)
        self.train_platform.upload_artifact(artifact_name, artifact_path)        

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        # self._anneal_lr()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        t, weights = self.schedule_sampler.sample(batch.shape[0], dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch,  # [bs, ch, image_size, image_size]
            t,  # [bs](int) sampled timesteps
            model_kwargs=cond,
            dataset=self.data.dataset
        )
        
        losses = compute_losses()


        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion.num_timesteps, t, {k: v * weights for k, v in losses.items()}
        )
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.resume_step)
        logger.logkv("samples", (self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.resume_step):09d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint(save_dir: str):
    matches = {file: re.match(r'model(\d+).pt', file) for file in os.listdir(save_dir)}
    models = {int(match.group(1)): file for file, match in matches.items() if match}
    
    return pjoin(save_dir, models[max(models)]) if models else None


def log_loss_dict(num_timesteps, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
