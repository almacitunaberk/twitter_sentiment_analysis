from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import lightning as l
import torch

class TensorsDataset(Dataset):
    def __init__(self, tokens, labels, is_test=False):
        super().__init__()
        self.tokens = tokens
        self.labels = labels
        self.is_test = is_test

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        if not self.is_test:
            return {
                "tokens": torch.from_numpy(self.tokens[index]),
                "targets": torch.tensor(self.labels[index], dtype=torch.long)
            }
        else:
            return {
                "tokens": torch.from_numpy(self.tokens[index])
            }


class TensorsDataModule(l.LightningDataModule):
    def __init__(self, tokens, labels, config):
        super().__init__()
        self.tokens = tokens
        self.labels = labels
        self.config = config

    def setup(self, stage):
        if stage == "fit":
            train_tokens, val_tokens, train_labels, val_labels = train_test_split(self.tokens, self.labels, test_size=0.2)
            self.train_data = TensorsDataset(tokens=train_tokens, labels=train_labels)
            self.val_data = TensorsDataset(tokens=val_tokens, labels=val_labels)
        if stage == "predict":
            self.test_data = TensorsDataset(tokens=self.tokens, labels=None, is_test=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            shuffle=True,
            batch_size=self.config.general.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            shuffle=True,
            batch_size=self.config.general.batch_size,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            shuffle=False,
            batch_size=self.config.general.batch_size
        )