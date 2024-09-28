from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

class PredictionsDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        super(PredictionsDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.config = config
        self.tweets = self.df["text"].values
        self.indices = self.df["indices"].values.astype("int64")
        print(f"Dataset indices length {len(self.indices)}")
        if not self.config.general.test_data:
            self.labels = self.df["labels"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = str(self.tweets[index])

        if len(tweet) == 0:
            tweet = "<eos>"

        inputs = self.tokenizer(
            tweet,
            max_length=100,
            padding="max_length",
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if not self.config.general.test_data:
            return {
                "input_ids": torch.tensor(ids,dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "indices": torch.tensor(self.indices[index], dtype=torch.int64),
                "index": torch.tensor(index, dtype=torch.long),
                "labels": torch.tensor(self.labels[index], dtype=torch.long)
            }
        else:
            return {
                "input_ids": torch.tensor(ids,dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "indices": torch.tensor(self.indices[index], dtype=torch.int64),
                "index": torch.tensor(index, dtype=torch.long),
            }