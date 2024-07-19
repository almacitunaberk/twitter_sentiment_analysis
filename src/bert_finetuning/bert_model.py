from transformers import DataCollatorWithPadding, get_scheduler, AutoTokenizer, AutoConfig, AutoModel
import transformers
import torch
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self, config):
        super(BERTModel, self).__init__()
        self.config = config
        self.bert_model_config = AutoConfig.from_pretrained(config.model.name, output_hidden_state=True)
        self.bert_model = AutoModel.from_pretrained(config.model.name, config=self.bert_model_config)
        #summary(self.bert_model, input_size=(32, 100, 768))
        self.dropout = nn.Dropout(config.model.dropout_prob)
        self.fc = nn.Linear(self.bert_model_config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        sequence_outputs = self.dropout(last_hidden_state)
        ## We only use the CLS token
        ## The first dimension is the batch size, the second dimension is the input tokens. The last dimension is the hidden representation size
        ## With :,0,: we get the vector corresponding to the CLS token
        cls_token = sequence_outputs[:, 0, :].view(-1, self.bert_model_config.hidden_size)
        logits = self.fc(cls_token)
        logits = torch.sigmoid(logits)
        return logits