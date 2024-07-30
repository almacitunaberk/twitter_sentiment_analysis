import argparse
from types import SimpleNamespace
import yaml
import pandas as pd
import os
import sys
import torch
import numpy as np
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
sys.path.append(os.path.join(filename, 'finetuning'))
from tokenizer import Tokenizer
import lightning as l
from lightning.pytorch import Trainer, seed_everything
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import resource
import wandb
import uuid
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from tqdm import tqdm
from ast import literal_eval
from ensemble_model_cls_tokens import EnsembleModel
from utils import get_namespace
from ensemble_data_cls_tokens import TensorsDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

def train(config, tokens, labels):
    seed_everything(config.general.seed)
    if config.general.use_wandb:
        wandb.login()

    run_id = config.general.run_id
    random_extension = str(uuid.uuid4())
    random_extension = random_extension[:6]
    log_dir = f"./saved_models/{run_id}_{random_extension}"
    if config.general.use_wandb:
        logger = WandbLogger(
            project="twitter_sent_analysis",
            name=run_id,
            entity="almacitunaberk-eth",
            save_dir=log_dir,
            log_model="all"
        )

    checkpoint_callb = ModelCheckpoint(
        monitor="val_loss",
        filename= '{epoch:02d}-{step}-{val_loss:.2f}',
        save_top_k=2,
        mode="min",
        dirpath=log_dir,
        enable_version_counter=True
    )

    callback_list = [checkpoint_callb]

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if accelerator in ["gpu", "cuda"]:
        print("Using the GPU")
    else:
        print("GPU is not being used")

    trainer = Trainer(
        accelerator="gpu",
        log_every_n_steps=1000,
        logger=logger if config.general.use_wandb else None,
        callbacks=callback_list,
        max_epochs = config.general.max_epochs,
        val_check_interval=0.5
    )

    data = TensorsDataModule(tokens=tokens, labels=labels, config=config)
    model = EnsembleModel(lr=config.model.lr)
    trainer.fit(model, data)

    if config.general.use_wandb:
        wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)

    config = get_namespace(config)
    print("Loading first tokens")
    tokens1 = np.load("/home/ubuntu/twitter_sentiment_analysis/src/cls_tokens/cardiff-base.npy", allow_pickle=True)
    print("Loaded first tokens")
    print("Loading second tokens")
    tokens2 = np.load("/home/ubuntu/twitter_sentiment_analysis/src/cls_tokens/vinai-base.npy", allow_pickle=True)
    print("Loaded second tokens")
    new_tokens = [np.empty(0, dtype=float) for i in range(len(tokens1))]
    for i, (token1, token2) in enumerate(zip(tokens1, tokens2)):
        new_tokens[i] = np.add(token1, token2)
    new_tokens = new_tokens[750000:1750000]
    print(len(new_tokens))
    print(len(new_tokens[0]))
    df = pd.read_csv("/home/ubuntu/data/raw_processed.csv")
    labels = df["labels"].values
    labels = labels[750000:1750000]
    print(len(labels))
    print("Starting training")
    train(config=config, tokens=new_tokens, labels=labels)


