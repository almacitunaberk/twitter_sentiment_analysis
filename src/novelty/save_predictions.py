import argparse
from types import SimpleNamespace
import yaml
import pandas as pd
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from predictions_models import PredictionsModel
from utils import get_namespace
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, AutoConfig
from predictions_models import PredictionsModel
from predictions_dataset import PredictionsDataset
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model_predictions(config, df):
    if not config.general.test_data:
        labels = df["labels"].values
    indices = df["indices"].values.astype("int64")
    preds = [np.empty(0, dtype=float) for i in range(len(df))]
    print(len(preds))
    model_save_path = config.model.save_path
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_model)
    model = PredictionsModel.load_from_checkpoint(model_save_path)
    model.to(device)
    model.freeze()
    model.eval()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data = PredictionsDataset(df=df, tokenizer=tokenizer, config=config)
    train_loader = DataLoader(train_data, batch_size=config.general.batch_size, shuffle=False, collate_fn=collator)

    with torch.no_grad():
        with tqdm(total=len(train_loader), leave=False) as pbar:
            for j, batch in enumerate(train_loader):
                batch = batch.to(device)
                outs = model(batch)
                outs = outs.to("cpu")
                for k, out in enumerate(outs):
                    index = int(batch["index"][k])
                    out = out.to("cpu").numpy()
                    _tensor = preds[index]
                    cat_tensor = np.concatenate((_tensor, out))
                    preds[index] = cat_tensor
                pbar.update(1)
            model.to("cpu")
            del model
            del collator
            del train_data
            del train_loader
    preds = np.array(preds)
    np.save(f"{config.general.cls_save_path}/{config.model.cls_filename}", preds)
    #pred_df = pd.DataFrame({"predictions": preds.tolist(), "label": labels, "index": indices})
    #pred_df.to_csv(config.general.df_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path")

    args = parser.parse_args()

    assert args.config_path is not None

    config = None
    with open(f"{args.config_path}", "r") as f:
        config = yaml.safe_load(f)

    config = get_namespace(config)
    df = pd.read_csv(config.general.df_path)
    df["indices"] = np.array([i for i in range(len(df))])
    print(df.info())
    print(len(df))
    print("Started converting predictions")
    save_model_predictions(config, df=df)
    print("Converted the predictions")
