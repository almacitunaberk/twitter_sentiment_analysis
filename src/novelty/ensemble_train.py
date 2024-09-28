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

    data = TensorsDataModule(tokens=tokens, labels=labels, config=config)

    for threshold in np.arange(0.3, 0.8, 0.05):
        if config.general.use_wandb:
            wandb.login()
        save_name = config.model.save_name
        save_name = f"{save_name}_threshold_{threshold}"
        
        if config.general.use_wandb:
            logger = WandbLogger(
                project=config.general.project,
                name=config.general.run_id,
                entity=config.general.entity,
                save_dir="./logs",
                log_model="all"
            )

        checkpoint_callb = ModelCheckpoint(
            monitor="val_acc",
            filename= save_name,
            save_top_k=1,
            mode="min",
            dirpath=config.model.save_path,
            enable_version_counter=True
        )

        callback_list = [checkpoint_callb]

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        if accelerator in ["gpu", "cuda"]:
            print("Using the GPU")
        else:
            print("GPU is not being used")

        trainer = Trainer(
            accelerator=accelerator,
            log_every_n_steps=100,
            logger=logger if config.general.use_wandb else None,
            callbacks=callback_list,
            max_epochs = config.general.max_epochs,
            val_check_interval=0.5
        )
        model = EnsembleModel(lr=config.model.lr, threshold=threshold)
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
    
    tokens = None
    
    for i, file in enumerate(os.listdir(config.general.tokens_folder)):
        if i == 0:
            tokens = np.load(f"{config.general.tokens_folder}/{file}", allow_pickle=True)
        else:
            _tokens = np.load(f"{config.general.tokens_folder}/{file}", allow_pickle=True)
            for i, (token1, token2) in enumerate(zip(_tokens, tokens)):
                tokens[i] = np.add(token1, token2)
    
    df = pd.read_csv(config.general.df_path)
    labels = df["labels"].values
    train(config=config, tokens=tokens, labels=labels)