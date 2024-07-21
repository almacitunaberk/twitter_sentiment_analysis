import lightning as l
from bert_finetune import Model
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

class PretrainedModel(l.LightningModule):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.backbone = Model.load_from_checkpoint(checkpoint_path)
        self.backbone.freeze()

    def forward(self, x):
        return self.backbone(x)

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(CustomDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.tweets = self.df["text"].values

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
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids,dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--test_path")
    parser.add_argument("--results_save_path")

    args = parser.parse_args()

    if args.checkpoint_path is None or args.test_path is None or args.results_save_path is None:
        print("Arguments cannot be None")
        exit()

    pretrained_model = PretrainedModel(args.checkpoint_path)

    if torch.cuda.is_available():
        pretrained_model.to(device)

    test_df = pd.read_csv(args.test_path)

    config = {
        "model": {
            "batch_size": 32,
            "name": "bert-base-uncased",

        }
    }
    #test_df = test_df.sample(n=32)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_data = CustomDataset(test_df, tokenizer=tokenizer)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collator)

    final_preds = []
    pretrained_model.eval()
    with tqdm(total=len(test_loader)) as pbar:
        for i, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k,v in batch.items()}
            with torch.no_grad():
                outputs = pretrained_model(batch)
            preds = (outputs >= 0.5).int()
            final_preds.extend(preds.cpu().numpy().tolist())
            pbar.update(1)
    pred_df = pd.DataFrame()
    pred_df["Id"] = np.arange(1, len(final_preds)+1)
    pred_df["Prediction"] = np.array(final_preds).ravel()
    pred_df["Prediction"] = pred_df["Prediction"].replace(0, -1)
    pred_df.to_csv(f"{args.results_save_path}/test_preds_1.csv", index=False)



