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
from ast import literal_eval
from ensemble_model import EnsembleModel

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

class CustomDataset(Dataset):
    def __init__(self, df):
        super(CustomDataset, self).__init__()
        self.df = df
        self.ids = self.df["index"].values
        self.predictions = self.df["predictions"].values


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            "tensors": torch.from_numpy(self.predictions[index]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)

    config = get_namespace(config)

    pretrained_model = EnsembleModel.load_from_checkpoint(config.model.save_path)

    if torch.cuda.is_available():
        pretrained_model.to(device)

    test_df = pd.read_csv(config.general.test_path)
    def makeArray(rawdata):
        string = literal_eval(rawdata)
        return np.array(string)

    test_df['predictions'] = test_df['predictions'].apply(lambda x: makeArray(x))

    test_data = CustomDataset(test_df)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    final_preds = []
    pretrained_model.eval()
    with tqdm(total=len(test_loader)) as pbar:
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = pretrained_model(batch["tensors"].float())
            preds = (outputs >= 0.5).int()
            final_preds.extend(preds.cpu().numpy().tolist())
            pbar.update(1)
    pred_df = pd.DataFrame()
    pred_df["Id"] = np.arange(1, len(final_preds)+1)
    pred_df["Prediction"] = np.array(final_preds).ravel()
    pred_df["Prediction"] = pred_df["Prediction"].replace(0, -1)
    pred_df.to_csv(f"{config.general.result_save_path}", index=False)


