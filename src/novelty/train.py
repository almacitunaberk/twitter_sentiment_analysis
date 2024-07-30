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
from ensemble_model import EnsembleModel
from utils import get_namespace
from ensemble_data import TensorsDataModule

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

def train(config, df):
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

    data = TensorsDataModule(df=df, config=config)
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
    train_df = pd.read_csv(config.general.train_path)
    # literal_eval converts the string to a list of tuples
    # np.array can convert this list of tuples directly into an array
    def makeArray(rawdata):
        string = literal_eval(rawdata)
        return np.array(string)

    # Applying the function row-wise, there could be a more efficient way
    train_df['predictions'] = train_df['predictions'].apply(lambda x: makeArray(x))
    print("Starting training")
    train(config=config, df=train_df)


