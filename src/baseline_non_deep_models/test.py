
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
from finetune_train import PLModel
import lightning as l


device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

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
        mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(ids,dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


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
    
    model_save_path = config.model.save_path

    test_df = pd.read_csv(config.general.test_path)

    train_df = pd.read_csv(config.general.train_path)
    train_df.dropna()

    embedding_type = config.model.embedding_type

    if embedding_type == "bow":
        vectorizer = CountVectorizer(max_features=config.model.embedding_max_features)
        vectorizer.fit_transform(train_df["text"].astype("U").values)
        X_test = vectorizer.transform(test_df["text"].astype("U").values)
    elif embedding_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=config.model.embedding_max_features)
        X_test = vectorizer.transform(tweets)
    elif embedding_type == "glove":
        glove = GloVe(name="twitter.27B", cache="/home/ubuntu/twitter_sentiment_analysis/.vector_cache", dim=config.model.embedding_dimension)
        tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True)
        X_test = np.array([np.mean(glove.get_vecs_by_tokens(tokenizer.tokenize_tweet(tweet=tweet), lower_case_backup=True).numpy(), axis=0) for tweet in tweets if len(tweet) != 0], dtype="float64")

    model_type = config.model.model_type

    if  model_type == "pkl":
        model = pickle.load(open(model_save_path, "rb"))
        preds = model.predict(X_test)
        preds[(preds == 0)] = -1

    if model_type == "ckpt":
        checkpoint_path = config.model.save_path
        model = PLModel.load_from_checkpoint(checkpoint_path)
        if torch.cuda.is_available():
            model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_model)
        collator = DataCollatorWithPadding(tokenizer=tokenizer)
        test_data = CustomDataset(test_df, tokenizer=tokenizer)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collator)

        trainer = l.Trainer(
            accelerator="gpu",
            enable_progress_bar=True,
            logger=False
        )
        test_outs = trainer.predict(model, test_loader)

        final_preds = []
        for out_stack in test_outs:
            for out in out_stack:
                if out.item() >= 0.5:
                    final_preds.append(1)
                else:
                    final_preds.append(-1)

        """
        model.eval()
        with tqdm(total=len(test_loader)) as pbar:
            for i, batch in enumerate(test_loader):
                batch = {k: v.to(device) for k,v in batch.items()}
                with torch.no_grad():
                    outputs = model(batch)
                preds = (outputs >= 0.5).int()
                final_preds.extend(preds.cpu().numpy().tolist())
                pbar.update(1)
        """
        pred_df = pd.DataFrame()
        pred_df["Id"] = np.arange(1, len(final_preds)+1)
        pred_df["Prediction"] = np.array(final_preds).ravel()
        pred_df.to_csv(f"{config.general.submission_path}/{config.model.submission_name}.csv", index=False)
    """
    ids = np.array([i+1 for i in range(len(test_df))])
    sub_df = pd.DataFrame({"Id": ids, "Prediction": preds})
    sub_df.to_csv(f"{config.general.submission_path}/{config.model.submission_name}.csv", index=False)
    """

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