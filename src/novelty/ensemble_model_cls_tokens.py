import torch.nn as nn
import lightning as l
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(768, 1)

    def forward(self, x):
        logits = self.fc1(x)
        logits = torch.sigmoid(logits)
        return logits

class EnsembleModel(l.LightningModule):
    def __init__(self, lr, threshold):
        super(EnsembleModel, self).__init__()
        self.save_hyperparameters()
        self.model = LinearModel()
        self.loss_fn = nn.BCELoss()
        self.lr = lr
        self.threshold = threshold

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tokens"].float())
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= self.threshold).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("train_loss", loss.item())
        self.log("train_acc", accuracy.item())
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["targets"]
        targets = targets.unsqueeze(1).float()
        outputs = self.model(batch["tokens"].float())
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= self.threshold).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("val_loss", loss.item(), on_step=True)
        self.log("val_acc", accuracy.item(), on_step=True)
        return loss

    def predict_step(self, batch, batch_idx):
        outputs = self.model(batch["tokens"].float())
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.00002, max_lr=0.001, cycle_momentum=False)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]