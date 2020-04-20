import os

import addict
import fire
import torch
import pandas as pd
from torch.backends import cudnn
from sklearn.model_selection import train_test_split

from . import getters
from . import training
from .training.config import parse_config, save_config
from .training.runner import GPUNormRunner

cudnn.benchmark = True


def worker_init_fn(seed):
    import random
    import numpy as np
    import time
    seed = (seed + 1) * (int(time.time()) % 60)  # set random seed every epoch!
    random.seed(seed + 1)
    np.random.seed(seed)


def main(cfg):
    # set GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus)) if cfg.get("gpus") else ""

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    print('Creating model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = getters.get_model(architecture=cfg.model.architecture, init_params=cfg.model.init_params)

    print('Moving model to device...')
    model.to(device)

    print('Collecting model parameters...')
    params = model.parameters()

    if len(cfg.gpus) > 1:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = torch.nn.DataParallel(model)
        model.to(device)

    # --------------------------------------------------
    # define datasets and dataloaders
    # --------------------------------------------------
    print('Creating datasets and loaders..')

    df = pd.read_csv(cfg.data.df_path)
    train_ids = df[df.fold != int(cfg.data.fold)].image_id.values
    valid_ids = df[df.fold == int(cfg.data.fold)].image_id.values

    assert (len(valid_ids)) != 0
    assert (len(train_ids)) != 0
    assert not set(train_ids).intersection(set(valid_ids))

    train_dataset = getters.get_dataset(
        name=cfg.data.train_dataset.name,
        init_params=cfg.data.train_dataset.init_params,
        ids=train_ids,
    )

    # add extra data (tier 2 or pseudo labels)
    for k, v in cfg.data.items():
        if k.startswith("dataset"):
            dataset = getters.get_dataset(name=v.name, init_params=v.init_params)
            print(f"Dataset `{k}`: {len(dataset)}")
            train_dataset = train_dataset + dataset

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **cfg.data.train_dataloader,
        worker_init_fn=worker_init_fn,
    )

    valid_dataset = getters.get_dataset(
        name=cfg.data.valid_dataset.name,
        init_params=cfg.data.valid_dataset.init_params,
        ids=valid_ids,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader
    )

    # --------------------------------------------------
    # define losses and metrics functions
    # --------------------------------------------------

    print('Defining losses and metrics..')
    losses = {}
    for output_name in cfg.training.losses.keys():
        loss_name = cfg.training.losses[output_name].name
        loss_init_params = cfg.training.losses[output_name].init_params
        losses[output_name] = getters.get_loss(loss_name, loss_init_params)

    metrics = {}
    for output_name in cfg.training.metrics.keys():
        metrics[output_name] = []
        for metric in cfg.training.metrics[output_name]:
            metrics[output_name].append(
                getters.get_metric(metric.name, metric.init_params)
            )

    # --------------------------------------------------
    # define optimizer and scheduler
    # --------------------------------------------------
    print('Defining optimizers and schedulers..')
    optimizer = getters.get_optimizer(
        cfg.training.optimizer.name,
        model_params=params,
        init_params=cfg.training.optimizer.init_params,
    )
    if cfg.training.get("scheduler", None):
        scheduler = getters.get_scheduler(
            cfg.training.scheduler.name,
            optimizer,
            cfg.training.scheduler.init_params,
        )
    else:
        scheduler = None

    # --------------------------------------------------
    # define callbacks
    # --------------------------------------------------
    print('Defining callbacks..')
    callbacks = []

    # add scheduler callback
    if scheduler is not None:
        callbacks.append(training.callbacks.Scheduler(scheduler))

    # add default logging and checkpoint callbacks
    if cfg.logdir is not None:
        # tb logging
        callbacks.append(training.callbacks.TensorBoard(
            os.path.join(cfg.logdir, 'tb')
        ))

        # checkpointing
        callbacks.append(training.callbacks.ModelCheckpoint(
            directory=os.path.join(cfg.logdir, 'checkpoints'),
            monitor="val_mask_" + metrics["mask"][0].__name__,
            save_best=True,
            save_top_k=0,
            mode="max",
            verbose=True,
        ))

    # --------------------------------------------------
    # start training
    # --------------------------------------------------
    print('Start training...')

    runner = GPUNormRunner(model, model_device=device)
    runner.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
    )

    runner.fit(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        callbacks=callbacks,
        **cfg.training.fit,
    )

if __name__ == "__main__":

    cfg = addict.Dict(fire.Fire(parse_config))
    logdir = cfg.get("logdir", None)
    if logdir is not None:
        save_config(cfg.to_dict(), logdir, name="config.yml")
        print(f"Config saved to: {logdir}")

    main(cfg)
    os._exit(0)
