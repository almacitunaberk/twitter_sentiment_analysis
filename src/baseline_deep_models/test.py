
from types import SimpleNamespace
import pandas as pd
from tqdm.auto import tqdm
import yaml
import os
import torch
import argparse
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

import sys
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
sys.path.append(os.path.join(filename, 'finetuning'))
from tokenizer import Tokenizer

from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
from bilstm_model import BiLSTMLightningModel
from bilstm_dataset import BiLSTMDataModule
import lightning as l


device = "gpu" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

def get_length(tweet):
    return len(tweet)

def get_namespace(data:dict):
    if type(data) is dict:
        new_ns = SimpleNamespace()
        for key, value in data.items():
            setattr(new_ns, key, get_namespace(value))
        return new_ns
    else:
        return data

def test(config):
    
    model_checkpoint = config.testing.model_checkpoint

    test_df = pd.read_csv(config.testing.test_input_path)

    model = BiLSTMLightningModel.load_from_checkpoint(model_checkpoint)
    if torch.cuda.is_available():
        model.to(device)

    tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)
    tokenized_tweets = [tokenizer.tokenize_tweet(tweet) for tweet in test_df["text"]]
    test_df = pd.DataFrame({"text": tokenized_tweets})
    test_data = BiLSTMDataModule(df=test_df, config=config)

    trainer = l.Trainer(
        accelerator=device,
        enable_progress_bar=True,
        logger=False
    )
    
    test_outs = trainer.predict(model, test_data)

    final_preds = []
    for out_stack in test_outs:
        for out in out_stack:
            if out.item() >= 0.5:
                final_preds.append(1)
            else:
                final_preds.append(-1)

    pred_df = pd.DataFrame()
    pred_df["Id"] = np.arange(1, len(final_preds)+1)
    pred_df["Prediction"] = np.array(final_preds).ravel()
    pred_df.to_csv(f"{config.testing.test_save_path}/{config.testing.test_filename}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)

    test(config)