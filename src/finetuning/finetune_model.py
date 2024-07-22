from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn

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
        """
        dropout_params = sum(p.numel() for p in self.dropout.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        print(dropout_params)
        print(fc_params)
        print(dropout_params + fc_params)
        """

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