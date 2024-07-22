import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.optim import AdamW
import torch.nn as nn
from finetune_model_mamba import df, config, BERTMAMBAModel


class FinetuneDataset(Dataset):
    def __init__(self, df, tokenizer, config, is_test=False):
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
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "targets": torch.tensor(self.labels[index], dtype=torch.long)
            }
        else:
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }


class FinetuneDataModule(pl.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.config = config
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.batch_size = config.model.batch_size

    def setup(self, stage: str):
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
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=self.config.model.dataloader_workers
        )


class SentimentAnalysisModule(pl.LightningModule):
    def __init__(self, model, lr=2e-5):
        super(SentimentAnalysisModule, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.model(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['targets']
        loss, logits = self.forward(input_ids, attention_mask, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['targets']
        loss, logits = self.forward(input_ids, attention_mask, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return optimizer


data_module = FinetuneDataModule(df, config)

model = BERTMAMBAModel(pretrained_model_name=config.model.name, num_labels=2)

pl_module = SentimentAnalysisModule(model)

trainer = pl.Trainer(
    max_epochs=config.model.num_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1 if torch.cuda.is_available() else None
)

trainer.fit(pl_module, data_module)
