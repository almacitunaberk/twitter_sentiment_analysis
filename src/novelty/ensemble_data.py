from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import lightning as l
import torch

class TensorsDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.predictions = self.df["predictions"].values
        self.labels = self.df["label"].values
        self.ids = self.df["index"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return {
            "tensors": torch.from_numpy(self.predictions[index]),
            "targets": torch.tensor(self.labels[index], dtype=torch.long)
        }


class TensorsDataModule(l.LightningDataModule):
    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def setup(self, stage):
        train_df, val_df = train_test_split(self.df, test_size=0.2)
        self.train_data = TensorsDataset(df=train_df)
        self.val_data = TensorsDataset(df=val_df)

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