import torch
import numpy as np
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from utils.fixseed import fixseed
from utils.parser_util import train_args


if __name__ == '__main__':
    args = train_args()
    fixseed(args.seed)

    logger = WandbLogger(project='mdm', 
                            id=args.experiment_name,
                            name=args.experiment_name,
                            resume='allow',
                            config={'batch_size': args.batch_size,
                                    'architecture': args.arch,
                                    'latent': args.latent_dim})

    model_cfg = get_config("configs/intergen.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml").interhuman

    datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    model = build_models(model_cfg)
    print('####### Loaded model successfully')

    checkpoint = find_last_checkpoint(args.experiment_name)

    litmodel = LitTrainModel(model, train_cfg, args.experiment_name)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH,
                                                       save_last=True)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices='auto', accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=2
    )

    trainer.fit(model=litmodel, datamodule=datamodule, ckpt_path=checkpoint)
