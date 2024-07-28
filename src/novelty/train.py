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

def mem():
    print('Memory usage         : % 2.2f MB' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0,1)
    )

mem()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(CustomDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values
        self.indices = self.df["indices"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.tweets[index]

        inputs = self.tokenizer(
            tweet,
            max_length=100,
            padding="max_length",
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids,dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "indices": torch.tensor(self.indices[index], dtype=torch.long)
        }

class FinetuneModel(nn.Module):
    def __init__(self, config):
        super(FinetuneModel, self).__init__()
        self.config = config
        self.base_model_config = AutoConfig.from_pretrained(config.model.name, output_hidden_state=True)
        self.base_model = AutoModel.from_pretrained(config.model.name, config=self.base_model_config)
        for param in self.base_model.parameters():
            param.requires_grad = self.config.model.base_model_require_grad
        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.fc1 = nn.Linear(self.base_model_config.hidden_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)
        """
        dropout_params = sum(p.numel() for p in self.dropout.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        print(dropout_params)
        print(fc_params)
        print(dropout_params + fc_params)
        """

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        ## We only use the CLS token
        ## The first dimension is the batch size, the second dimension is the input tokens. The last dimension is the hidden representation size
        ## With :,0,: we get the vector corresponding to the CLS token
        cls_token = sequence_outputs[:, 0, :].view(-1, self.base_model_config.hidden_size)
        """
        logits = self.fc1(cls_token)
        logits = self.dropout(logits)
        logits = self.relu1(logits)
        logits = self.fc2(logits)
        logits = self.dropout(logits)
        logits = self.relu2(logits)
        logits = self.fc3(logits)
        logits = torch.sigmoid(logits)
        """
        return cls_token

class PLModel(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = FinetuneModel(config=config)
        self.loss_fn = nn.BCELoss()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x["input_ids"], x["attention_mask"]), x["indices"]

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
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.001)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


def get_namespace(data):
    if type(data) is dict:
        new_ns = SimpleNamespace()
        for key, value in data.items():
            setattr(new_ns, key, get_namespace(value))
        return new_ns
    else:
        return data


def get_tokens(config, train_df):

    train_df = train_df.head(1000) # TODO: Comment here on AWS
    for t in range(0, len(train_df), 32):
        truncated_df = train_df.iloc[t:t+32]
        for i in range(4):
            model_save_path = None
            if i == 0:
                model_save_path = config.model1.save_path
                tokenizer = AutoTokenizer.from_pretrained(config.model1.backbone_model)
            if i == 1:
                model_save_path = config.model2.save_path
                tokenizer = AutoTokenizer.from_pretrained(config.model2.backbone_model)
            if i == 2:
                model_save_path = config.model3.save_path
                tokenizer = AutoTokenizer.from_pretrained(config.model3.backbone_model)
            if i == 3:
                model_save_path = config.model4.save_path
                tokenizer = AutoTokenizer.from_pretrained(config.model4.backbone_model)

            model = PLModel.load_from_checkpoint(model_save_path)
            model.to(device)
            model.eval()

            collator = DataCollatorWithPadding(tokenizer=tokenizer)
            train_data = CustomDataset(truncated_df, tokenizer=tokenizer)
            train_loader = DataLoader(train_data, batch_size=32, shuffle=False, collate_fn=collator)

            with torch.no_grad():
                with tqdm(total=len(train_loader), leave=False) as pbar:
                    for j, batch in enumerate(train_loader):
                        batch.to(device)
                        outs, indices = model(batch)
                        outs.to(device)
                        for k, (out, index) in enumerate(zip(outs, indices)):
                            if i == 0:
                                torch.save(out, f"tensors/tensor_{index}.pt")
                            else:
                                tensor = torch.load(f"tensors/tensor_{index}.pt")
                                concatenated = torch.cat((tensor, out))
                                torch.save(concatenated, f"tensors/tensor_{index}.pt")
                        pbar.update(1)
                    model.to("cpu")
                    del model
                    del collator
                    del train_data
                    del train_loader

def train(config, df):
    seed_everything(config.general.seed)
    if config.general.use_wandb:
        wandb.login()

    run_id = config.general.run_id
    random_extension = str(uuid.uuid4())
    random_extension = random_extension[:6]
    log_dir = f"./saved_models/{run_id}_{random_extension}"
    if config.general.use_wandb:
        logger = WandbLogger(
            project="twitter_sent_analysis",
            name=run_id,
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

    accelerator = "gpu" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if accelerator in ["gpu", "cuda"]:
        print("Using the GPU")
    else:
        print("GPU is not being used")

    trainer = Trainer(
        accelerator="gpu",
        log_every_n_steps=1,
        logger=logger if config.general.use_wandb else None,
        callbacks=callback_list,
        max_epochs = config.general.max_epochs,
        val_check_interval=0.5
    )

    data = TensorsDataModule(folder="tensors", df=df, config=config)
    model = EnsembleModel()
    trainer.fit(model, data)

    if config.general.use_wandb:
        wandb.finish()

class TensorsDataset(Dataset):
    def __init__(self, folder, df):
        super().__init__()
        self.folder = folder
        self.df = df
        self.tweets = self.df["text"].values
        self.labels = self.df["labels"].values
        self.ids = self.df["indices"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            "tensors": torch.load(f"tensors/tensor_{self.ids[index]}.pt"),
            "targets": torch.tensor(self.labels[index], dtype=torch.long)
        }

class TensorsDataModule(l.LightningDataModule):
    def __init__(self, folder, df, config):
        super().__init__()
        self.folder = folder
        self.df = df
        self.config = config

    def setup(self, stage):
        train_df, val_df = train_test_split(self.df, test_size=0.2)
        self.train_data = TensorsDataset(df=train_df, folder=self.folder)
        self.val_data = TensorsDataset(df=val_df, folder=self.folder)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=False,
            batch_size=self.config.general.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=False,
            batch_size=self.config.general.batch_size,
        )
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(3584, 1792)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1792, 896)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(896, 448)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(448,224)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(224, 1)

    def forward(self, x):
        logits = self.fc1(x)
        logits = self.relu1(logits)
        logits = self.fc2(logits)
        logits = self.relu2(logits)
        logits = self.fc3(logits)
        logits = self.relu3(logits)
        logits = self.fc4(logits)
        logits = self.relu4(logits)
        logits = self.fc5(logits)
        logits = torch.sigmoid(logits)
        return logits

class EnsembleModel(l.LightningModule):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.model = LinearModel()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tensors"])
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("train_loss", loss.item())
        self.log("train_acc", accuracy.item())

    def validation_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tensors"])
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("val_loss", loss.item())
        self.log("val_acc", accuracy.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)

    config = get_namespace(config)
    train_df = pd.read_csv(config.general.train_path)
    train_df["indices"] = np.array([i for i in range(len(train_df))])
    train_df = train_df.dropna()
    train_df.to_csv("sample.csv", index=False)
    print("Started writing tensors")
    get_tokens(config, train_df) # Toggle this if you already have the tensors
    print("Wrote tensors")
    #train(config=config, df=train_df)


