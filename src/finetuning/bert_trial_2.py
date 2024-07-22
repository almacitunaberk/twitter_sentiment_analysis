import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import argparse
import torch.nn as nn
import transformers

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(CustomDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values
        self.labels = self.df["labels"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.tweets[index]

        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            padding="max_length",
            add_special_tokens=True,
            return_attention_mask=True,
            #max_length=self.max_length,
            truncation=True
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor(self.labels[index], dtype=torch.long)
        }

def train(df):
    training, val = train_test_split(df, test_size=0.2)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CustomDataset(training, tokenizer)
    val_dataset = CustomDataset(val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64)
    val_loader = DataLoader(val_dataset, batch_size=64)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.to(device)
    loss_fn = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    model.train()
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(2):
        print("Epoch {}".format(epoch+1))
        train_loss = 0.0
        num_correct = 0
        num_samples = 0
        for i, data in enumerate(train_loader):
            ids = data["ids"]
            ids = ids.to(device)
            token_type_ids = data["token_type_ids"]
            token_type_ids = token_type_ids.to(device)
            mask = data["mask"]
            mask = mask.to(device)
            targets = data["target"]
            targets = targets.to(device)
            targets = targets.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            targets = targets.as_type(outputs)
            preds = outputs.logits
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.sum().item()
            predictions = (outputs >= 0.5).int()
            num_samples += targets.size(0)
            num_correct += (predictions == targets).sum().item()
        train_accs.append(num_correct / num_samples)
        train_losses.append(train_loss / num_samples)

        model.eval()
        val_loss = 0.0
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                ids = data["ids"]
                ids = ids.to(device)
                token_type_ids = data["token_type_ids"]
                token_type_ids = token_type_ids.to(device)
                mask = data["mask"]
                mask = mask.to(device)
                targets = data["target"]
                targets = targets.to(device)
                targets = targets.unsqueeze(1)
                outputs = model(input_ids=ids, attention_mask=mask)
                targets = targets.as_type(outputs)
                preds = outputs.logits
                loss = loss_fn(preds, targets)
                val_loss += loss.sum().item()
                predictions = (outputs >= 0.5).int()
                num_samples += targets.size(0)
                num_correct += (predictions == targets).sum().item()
        val_accs.append(num_correct / num_samples)
        val_losses.append(val_loss / num_samples)
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
    df = df.sample(n=100000)
    train(df)