from torch.utils.data import Dataset, DataLoader
import torch
import lightning as l
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding

class FinetuneDataModule(l.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.config = config
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.batch_size = config.model.batch_size

    def setup(self, stage:str):
        train_df, val_df = train_test_split(self.df, test_size=self.config.general.validation_size)
        self.train_data = FinetuneDataset(df=train_df, tokenizer=self.tokenizer, config=self.config, is_test=False)
        self.val_data = FinetuneDataset(df=val_df, tokenizer=self.tokenizer, config=self.config, is_test=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.config.model.dataloader_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.config.model.dataloader_workers
        )

class FinetuneDataset(Dataset):
    def __init__(self, df, tokenizer, config, is_test):
        super(FinetuneDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values
        self.is_test = is_test
        if not self.is_test:
            self.labels = self.df["labels"].values
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.tweets[index]

        inputs = self.tokenizer(
            tweet,
            max_length=self.config.model.max_length,
            padding="max_length",
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        if not self.is_test:
            return {
                "input_ids": torch.tensor(ids,dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "targets": torch.tensor(self.labels[index], dtype=torch.long)
            }
        else:
            return {
                "input_ids": torch.tensor(ids,dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
