import torch
import numpy as np
import random
import os
import yaml
import argparse
import wandb
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from torch.utils.data import DataLoader
import lightning as l
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, BatchSizeFinder, EarlyStopping, DeviceStatsMonitor, LearningRateFinder, StochasticWeightAveraging
from finetune_dataset import FinetuneDataset, FinetuneDataModule
from finetune_model import FinetuneModel, FinetuneLightningModule
import uuid

def get_namespace(data:dict):
    if type(data) is dict:
        new_ns = SimpleNamespace()
        for key, value in data.items():
            setattr(new_ns, key, get_namespace(value))
        return new_ns
    else:
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the config file. This value cannot be null")
    args = parser.parse_args()
    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)
    seed_everything(config.general.rand_seed, workers=True)

    if config.general.use_wandb:
        wandb.login()
    
    model_save_name = config.model.save_name
    log_dir = f"../saved_models/"
    if config.general.use_wandb:
        logger = WandbLogger(
            project=config.general.project,
            name=config.general.run_id,
            entity=config.general.entity,
            save_dir=log_dir,
            log_model="all"
        )

    checkpoint_callb = ModelCheckpoint(
        monitor="val_loss",
        filename= model_save_name,
        save_top_k=1,
        mode="min",
        dirpath=log_dir,
        enable_version_counter=True
    )

    if config.general.use_model_summary:
        summary_callb = ModelSummary(
            max_depth=1
        )


    if config.model.use_batchsize_finder:
        batch_size_callb = BatchSizeFinder(
            mode="binsearch",
            init_val=256,
            max_trials=3,
        )

    if config.general.use_early_stopping:
        early_stopping_callb = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=5,
            mode="min"
        )

    if config.model.use_lr_finder:
        lr_finder_callb = LearningRateFinder(
            min_lr=0.0000001,
            max_lr=0.1,
            mode="exponential"
        )

    if config.general.use_swa:
        swa_callb = StochasticWeightAveraging(
            swa_lrs=0.0001
        )

    if config.model.use_device_stats:
        device_stats_callb = DeviceStatsMonitor(cpu_stats=False)

    callback_list = [checkpoint_callb]

    if config.model.use_batchsize_finder:
        callback_list.append(batch_size_callb)

    if config.model.use_lr_finder:
        callback_list.append(lr_finder_callb)

    if config.model.use_device_stats:
        callback_list.append(device_stats_callb)

    if config.general.use_model_summary:
        callback_list.append(summary_callb)

    if config.general.use_early_stopping:
        callback_list.append(early_stopping_callb)

    if config.general.use_swa:
        callback_list.append(swa_callb)

    df = pd.read_csv(config.general.data_path)

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if accelerator in ["gpu", "cuda"]:
        print("Using the GPU")
    else:
        print("GPU is not being used")

    if config.general.local:
        df = df.sample(n=config.general.reduced_length)

    trainer = Trainer(
        accelerator=accelerator,
        log_every_n_steps=1,
        logger=logger if config.general.use_wandb else None,
        callbacks=callback_list,
        strategy="auto",
        max_epochs = config.model.max_epochs,
        fast_dev_run = 5 if config.general.local else False,
        profiler=config.general.use_profiler,
        deterministic=True,
        val_check_interval=0.5
    )
    data = FinetuneDataModule(df=df, config=config)
    model = FinetuneLightningModule(config)
    trainer.fit(model, data)

    if config.general.use_wandb:
        wandb.finish()