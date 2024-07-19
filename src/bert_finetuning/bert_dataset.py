from torch.utils.data import Dataset, DataLoader
import torch
import lightning as l
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding

class CustomDataModule(l.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.config = config
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def setup(self, stage:str):
        train_df, val_df = train_test_split(self.df, test_size=self.config.general.validation_size)
        self.train_data = CustomDataset(df=train_df, tokenizer=self.tokenizer, config=self.config)
        self.val_data = CustomDataset(df=val_df, tokenizer=self.tokenizer, config=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.config.model.batch_size,
            collate_fn=self.collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=True,
            batch_size=self.config.model.batch_size,
            collate_fn=self.collator
        )



class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, config):
        super(CustomDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values
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
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.labels[index], dtype=torch.long)
        }
