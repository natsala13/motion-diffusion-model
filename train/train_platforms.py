import os
import wandb

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
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
        wandb.init(
                project='mdm',
                id=experiment_name,
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=experiment_name,
                resume='allow',
                config=config)

    def report_scalar(self, name, value, iteration, group_name=None):
        wandb.log({name: value}, step=iteration)

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

    def close(self):
        wandb.finish()
