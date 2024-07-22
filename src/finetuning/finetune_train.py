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
from finetune_model import FinetuneModel

class PLModel(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = FinetuneModel(config=config)
        self.loss_fn = nn.BCELoss()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x["input_ids"], x["attention_mask"])

    def training_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("train_loss", loss.item())
        self.log("train_acc", accuracy.item())
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("val_loss", loss.item())
        self.log("val_acc", accuracy.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.model.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.001)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


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
    parser.add_argument("--run_id", help="Run ID is needed to identify the Wandb run")
    parser.add_argument("--model_extension", help="If you added any extension to the base model, please identify it with an extension name", default="1")
    args = parser.parse_args()
    if args.config_path is None or args.run_id is None:
        print("Config_path or run_id arguments cannot be null")
        exit()
    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)
    seed_everything(config.general.rand_seed, workers=True)
    ## TODO: Comment this line when running on cluster
    if config.general.use_wandb:
        wandb.login()
    log_dir = f"./saved_models/{config.model.name}_{args.model_extension}"
    if config.general.use_wandb:
        logger = WandbLogger(
            project="twitter_sent_analysis",
            name=args.run_id,
            entity="almacitunaberk-eth",
            save_dir=log_dir,
            log_model="all"
        )

    checkpoint_callb = ModelCheckpoint(
        monitor="val_loss",
        filename= '{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
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
        strategy="ddp" if torch.cuda.is_available() else "auto",
        #check_val_every_n_epoch=10,
        max_epochs = config.model.max_epochs,
        fast_dev_run = 5 if config.general.local else False,
        profiler=config.general.use_profiler,
        deterministic=True,
    )

    data = FinetuneDataModule(df=df, config=config)
    model = PLModel(config)
    trainer.fit(model, data)

    if config.general.use_wandb:
        wandb.finish()