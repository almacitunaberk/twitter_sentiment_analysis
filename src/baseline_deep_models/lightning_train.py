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
wandb.login()
import numpy as np

def get_namespace(data):
    if type(data) is dict:
        new_ns = SimpleNamespace()
        for key, value in data.items():
            setattr(new_ns, key, get_namespace(value))
        return new_ns
    else:
        return data

def train(config, df):
    random_extension = str(uuid.uuid4())
    random_extension = random_extension[:6]
    log_dir = f"{config.general.log_path}_{random_extension}"

    if config.general.use_wandb:
        logger = WandbLogger(
            project="twitter_sent_analysis",
            name=config.general.run_id,
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
    trainer = Trainer(
        accelerator="gpu",
        logger=logger if config.general.use_wandb else None,
        callbacks=callback_list,
        #check_val_every_n_epoch=10,
        max_epochs = config.general.max_epochs,
        fast_dev_run = 5 if config.general.debug else False,
        deterministic=True,
        val_check_interval=0.5
    )
    #tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)
    tokenized_tweets = np.load("trial_np_save.npy", allow_pickle=True)
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
    train_df = pd.read_csv(config.general.input_path)
    train_df = train_df.dropna()
    print("Starting training")
    train(config=config, df=train_df)


