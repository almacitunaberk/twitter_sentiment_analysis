from torch.utils.data import Dataset, DataLoader
import torch
import lightning as l
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
import sys
import os
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
from torchtext.vocab import GloVe
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence

import numpy as np

def get_length(tweet):
    return len(tweet)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_batch = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_batch, labels

"""
def load_glove_embeddings(glove_path: str):
    print("Loading GloVe embeddings")
    embeddings = {}
    with open(glove_path, "r") as f:
        for line in f:
            word, weights = line.split(maxsplit=1)
            weights = np.fromstring(weights, "f", sep=" ")
            embeddings[word] = weights
    print("Loaded GloVe embeddings")
    return embeddings
"""

def tweet_embed(words, embeddings, glove_dim):
    unknown_indices = []
    mean = np.zeros(glove_dim)
    for i in range(len(words)):
        if words[i] in embeddings.keys():
            words[i] = embeddings[words[i]]
            mean += words[i]
        else:
            unknown_indices.append(i)
    if (len(words) - len(unknown_indices)) != 0:
        mean /= (len(words) - len(unknown_indices))
    for i in unknown_indices:
        words[i] = mean
    return np.array(words)

class BiLSTMDataModule(l.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def setup(self, stage:str):
        """
        print("Transforming tweets")
        self.df["text"] = self.df["text"].apply(lambda x: self.tokenizer.tokenize_tweet(x))
        np.save("trial_np_save", self.df["text"])
        print(type(self.df.iloc[0]["text"]))
        print(self.df.iloc[0]["text"])
        print("Transformed tweets")
        print("Removing length 0 tweets")
        self.df["length"] = self.df["text"].apply(get_length)
        self.df = self.df[self.df["length"] != 0]
        print("Removed zero length tweets")
        """
        train_df, val_df = train_test_split(self.df, test_size=self.config.general.validation_size)
        self.train_data = BiLSTMDataset(df=train_df, config=self.config)
        self.val_data = BiLSTMDataset(df=val_df, config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.config.general.batch_size,
            num_workers=self.config.general.dataloader_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.config.general.batch_size,
            num_workers=self.config.general.dataloader_workers
        )

class BiLSTMDataset(Dataset):
    def __init__(self, df, config):
        self.tweets = df["text"].to_list()
        self.labels = df["labels"].to_list()
        self.config = config
        self.glove = GloVe(name="twitter.27B", dim=self.config.model.glove_dim)

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        input_ids = self.glove.get_vecs_by_tokens(self.tweets[index]).view(-1)
        return input_ids, self.labels[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }