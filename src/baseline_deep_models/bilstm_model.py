import torch
import torch.nn as nn
import lightning as l

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BiLSTM(nn.Module):
    def __init__(self, embed_dim, drop_prob):
        super().__init__()
        self.hidden_size = 100
        self.input_size = embed_dim
        self.num_layers = 1
        self.bidirectional = True
        self.num_directions = 1
        self.dropout1 = nn.Dropout(p=drop_prob)

        if self.bidirectional:
            self.num_directions = 2

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.fc = nn.Linear(self.hidden_size*self.num_directions, 1)

    def forward(self, tweet):
        tweet = tweet.reshape(tweet.shape[0], tweet.shape[1] // self.input_size, self.input_size)
        lstm_out, _ = self.lstm(tweet)
        #x = self.dropout1(lstm_out.view(len(tweet), -1))
        output = self.fc(lstm_out[:, -1, :])
        pred = torch.sigmoid(output)
        return pred

class BiLSTMLightningModel(l.LightningModule):
    def __init__(self, config):
        super(BiLSTMLightningModel, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = BiLSTM(embed_dim=config.model.glove_dim, drop_prob=config.model.drop_prob)
        self.loss_fn = nn.BCELoss()

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.required_grad)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = torch.tensor(targets).float().unsqueeze(1).to(device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("train_loss", loss.item())
        self.log("train_acc", accuracy.item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = torch.tensor(targets).float().unsqueeze(1).to(device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = (outputs >= 0.5).float().detach()
        accuracy = (preds == targets).float().mean()
        self.log("val_loss", loss.item())
        self.log("val_acc", accuracy.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.model.lr)
        return optimizer
