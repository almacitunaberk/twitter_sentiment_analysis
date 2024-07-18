from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super(CustomDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values
        self.targets = self.df["labels"].values
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.tweets[index]

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.targets[index], dtype=torch.long)
        }

class CustomBERT(nn.Module):
    def __init__(self):
        super(CustomBERT, self).__init__()
        self.base_bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.base_bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        out = self.fc(o2)
        pred = torch.sigmoid(out)
        return pred

def train(df):
    training, val = train_test_split(df, test_size=0.2)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CustomDataset(training, tokenizer, max_length=100)
    val_dataset = CustomDataset(val, tokenizer, max_length=100)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32)
    model = CustomBERT()
    model.to(device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    for param in model.base_bert_model.parameters():
        param.requires_grad = False

    model.train()
    train_losses = []
    train_accs = []
    val_losses =[]
    val_accs = []
    for epoch in range(4):
        print(f"Epoch {epoch+1}")
        num_correct = 0
        num_samples = 0
        train_loss = 0
        for batch, dl in tqdm(enumerate(train_dataloader), leave=False, total=len(train_dataloader)):
            ids = dl["ids"]
            ids = ids.to(device)
            token_type_ids = dl["token_type_ids"]
            token_type_ids = token_type_ids.to(device)
            mask = dl["mask"]
            mask = mask.to(device)
            label = dl["target"]
            label = label.to(device)
            label = label.unsqueeze(1)
            optimizer.zero_grad()
            output = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            label = label.type_as(output)
            loss = loss_fn(output, label)
            train_loss += loss.sum().item()
            loss.backward()
            optimizer.step()
            preds = (output >= 0.5).int()
            num_correct += sum(1 for a, b in zip(preds, label) if a[0] == b[0])
            num_samples += preds.shape[0]
        train_accs.append(num_correct / num_samples)
        train_losses.append(train_loss / num_samples)

        val_loss = 0
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for batch, dl in tqdm(enumerate(val_dataloader), leave=False, total=len(val_dataloader)):
                ids = dl["ids"]
                ids = ids.to(device)
                token_type_ids = dl["token_type_ids"]
                token_type_ids = token_type_ids.to(device)
                mask = dl["mask"]
                mask = mask.to(device)
                label = dl["target"]
                label = label.to(device)
                label = label.unsqueeze(1)
                optimizer.zero_grad()
                output = model(
                    ids=ids,
                    mask=mask,
                    token_type_ids=token_type_ids
                )
                label = label.type_as(output)
                loss = loss_fn(output, label)
                val_loss += loss.sum().item()
                num_correct += sum(1 for a,b in zip(preds, label) if a[0] == b[0])
                num_samples += preds.shape[0]
        val_losses.append(val_loss/num_samples)
        val_accs.append(num_correct / num_samples)
        print("Epoch summary")
        print(f'Train Loss: {train_losses[-1]:7.2f}  Train Accuracy: {train_accs[-1]*100:6.3f}%')
        print(f'Validation Loss: {val_losses[-1]:7.2f}  Validation Accuracy: {val_accs[-1]*100:6.3f}%')
        print('')
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    args = parser.parse_args()
    assert args.input_dir is not None
    df = pd.read_csv(args.input_dir)
    train(df)