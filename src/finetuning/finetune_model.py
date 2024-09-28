from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn
import lightning as l

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

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        ## We only use the CLS token
        ## The first dimension is the batch size, the second dimension is the input tokens. The last dimension is the hidden representation size
        ## With :,0,: we get the vector corresponding to the CLS token
        cls_token = sequence_outputs[:, 0, :].view(-1, self.base_model_config.hidden_size)
        logits = self.fc1(cls_token)
        logits = self.dropout(logits)
        logits = self.relu1(logits)
        logits = self.fc2(logits)
        logits = self.dropout(logits)
        logits = self.relu2(logits)
        logits = self.fc3(logits)
        logits = torch.sigmoid(logits)
        return logits

class FinetuneLightningModule(l.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = FinetuneModel(config=config)
        self.loss_fn = nn.BCELoss()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x["input_ids"], x["attention_mask"])

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

    def predict_step(self, batch, batch_idx):
        outputs = self.model(batch["input_ids"], batch["attention_mask"])
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.model.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.001)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
