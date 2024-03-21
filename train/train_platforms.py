import os
from typing import Optional

import wandb
from wandb.wandb_run import Run

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass

    @property
    def config(self):
        return {}
    
    @property
    def name(self):
        return None
    
    def upload_artifact(self, *args, **kwargs):
        pass
    


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    pass


class WandBPlatform(TrainPlatform):
    def __init__(self, save_dir, experiment_name='mdm couple experiment', **config):
        print(f'Starting  experiment {experiment_name}')
        super().__init__(save_dir)
        wandb.login(key='deca9d4d3521670701b2b00b34124d8786ddc7fc')
        self.run= wandb.init(project='mdm',
                            id=experiment_name,
                            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                            name=experiment_name,
                            resume='allow',
                            config=config)

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log({name: value}, step=iteration)
    
    def upload_artifact(self, name: str, file_path: str):
        artifact = wandb.Artifact(name, type='motion')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)
    
    def report_args(self, args, name):
        pass
        # api = wandb.Api()

        # # Access attributes directly from the run object 
        # # or from the W&B App 
        # username = wandb.run.entity
        # project = wandb.run.project
        # run_id = wandb.run.id

        # run = api.run(f"{username}/{project}/{run_id}")
        # run.config["bar"] = 32
        # run.update()

    def close(self, exit_code: Optional[int]=None):
        wandb.finish(exit_code=exit_code)

    def close_preempting(self):
        wandb.mark_preempting()
        self.close(exit_code=1)


class WandBSweepPlatform(TrainPlatform):
    def __init__(self, save_dir, experiment_name='mdm couple experiment', **_):
        print(f'Starting  experiment {experiment_name}')
        super().__init__(save_dir)
        wandb.login(key='deca9d4d3521670701b2b00b34124d8786ddc7fc')
        self.run = wandb.init(project='mdm',
                                resume='allow')
        assert self.run is not None
        self.run.mark_preempting()

    @property
    def config(self):
        return self.run.config
    
    @property
    def name(self):
        return self.run.name

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log({name: value}, step=iteration)

    def report_args(self, args, name):
        pass
    
    def upload_artifact(self, name: str, file_path: str):
        artifact = wandb.Artifact(name, type='motion')
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)

    def close(self, exit_code: Optional[int]=None):
        wandb.finish(exit_code=exit_code)

    def close_preempting(self):
        wandb.mark_preempting()
        self.close(exit_code=1)

