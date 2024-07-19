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
import sys
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'bert_finetuning'))
from lightning.pytorch.callbacks import ModelCheckpoint
from bert_dataset import CustomDataset, CustomDataModule
from bert_model import BERTModel

wandb.login()

class Model(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = BERTModel(config=config)
        self.loss_fn = nn.BCELoss()
        for param in self.model.bert_model.parameters():
            param.required_grad = False
        self.model.bert_model.eval()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        return optimizer


def set_seed(seed):
    """
    Sets random number generator seeds for PyTorch and NumPy to ensure reproducibility of results.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    if args.config_path is None:
        print("Config_dir argument cannot be null")
        exit()
    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)
    set_seed(config.general.rand_seed)
    ## TODO: Comment this line when running on cluster
    config.general.data_path = "/Users/tunaberkalmaci/Downloads/twitter_sentiment_analysis/src/data/processed/raw_processed.csv"

    logger = WandbLogger(
        project="twitter_sent_analysis",
        name="bert_finetune",
        entity="almacitunaberk-eth")

    checkpoint_callb = ModelCheckpoint(
        monitor="val_loss",
        filename= "bert_finetune_epoch{epoch:02d}-val_loss{val_loss:.2f}",
        save_top_k=2,
        mode="min"
    )

    if config.general.testing:

        df = pd.read_csv(config.general.data_path)

        if config.general.reduced_samples:
            df = df.sample(n=config.general.reduced_length)

        callback_list = [checkpoint_callb]

        accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        trainer = l.Trainer(
            accelerator="cpu",
            log_every_n_steps=1,
            logger=logger,
            enable_checkpointing=True,
            callbacks=callback_list
        )
        data = CustomDataModule(df=df, config=config)
        model = Model(config)
        trainer.fit(model, data)
        wandb.finish()