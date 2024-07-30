import torch.nn as nn
import lightning as l
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.avg_pool = nn.AvgPool1d(4)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        """
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1792, 896)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(896, 448)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(448,224)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(224, 1)
        """

    def forward(self, x):
        logits = self.avg_pool(x)
        logits = torch.sub(logits, self.threshold)
        logits = torch.sigmoid(logits)
        return logits
        """
        logits = self.relu1(logits)
        logits = self.fc2(logits)
        logits = self.relu2(logits)
        logits = self.fc3(logits)
        logits = self.relu3(logits)
        logits = self.fc4(logits)
        logits = self.relu4(logits)
        logits = self.fc5(logits)
        """

class EnsembleModel(l.LightningModule):
    def __init__(self, lr):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.model = LinearModel()
        self.loss_fn = nn.BCELoss()
        self.lr = lr

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tensors"].float())
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("train_loss", loss.item())
        self.log("train_acc", accuracy.item())

    def validation_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tensors"].float())
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("val_loss", loss.item())
        self.log("val_acc", accuracy.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer