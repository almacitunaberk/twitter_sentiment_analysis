import argparse
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from bilstm_model import BiLSTMLightningModel
from bilstm_dataset import BiLSTMDataModule
from lightning.pytorch.loggers import WandbLogger
import sys
import os
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
from tokenizer import Tokenizer
from types import SimpleNamespace
import pandas as pd
import uuid
import wandb
import numpy as np
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"

def get_namespace(data):
    if type(data) is dict:
        new_ns = SimpleNamespace()
        for key, value in data.items():
            setattr(new_ns, key, get_namespace(value))
        return new_ns
    else:
        return data

def train(config, df):
    model_save_path = "../saved_models"
    logdir = "./logs"
    if config.general.use_wandb:
        logger = WandbLogger(
            project=config.general.wandb_project,
            name=config.general.run_id,
            entity=config.general.wandb_entity,
            save_dir=logdir,
            log_model="all"
        )

    checkpoint_callb = ModelCheckpoint(
        monitor="val_loss",
        filename= config.model.save_name,
        save_top_k=1,
        mode="min",
        dirpath=model_save_path,
        enable_version_counter=True
    )

    callback_list = [checkpoint_callb]
    trainer = Trainer(
        accelerator=device,
        logger=logger if config.general.use_wandb else None,
        callbacks=callback_list,
        max_epochs = config.general.max_epochs,
        fast_dev_run = 5 if config.general.debug else False,
        deterministic=True,
        val_check_interval=0.5
    )
    tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)
    tokenized_tweets = [tokenizer.tokenize_tweet(tweet) for tweet in df["text"]]
    lengths = [len(tweet) for tweet in tokenized_tweets]
    tokenized_df = pd.DataFrame({"text": tokenized_tweets, "labels": df["labels"].values, "length": lengths})
    tokenized_df = tokenized_df[tokenized_df["length"] != 0]
    data = BiLSTMDataModule(df=tokenized_df, config=config)
    model = BiLSTMLightningModel(config)
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
    if config.general.use_wandb:
        wandb.login()
    train_df = pd.read_csv(config.general.input_path)
    train_df = train_df.dropna()
    print("Starting training")
    train(config=config, df=train_df)


