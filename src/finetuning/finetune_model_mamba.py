import pandas as pd
from transformers import BertModel
from torch import nn


pos_file = 'train_pos_full.txt'
neg_file = 'train_neg_full.txt'

with open(pos_file, 'r') as f:
    pos_tweets = f.readlines()

with open(neg_file, 'r') as f:
    neg_tweets = f.readlines()

data = {
    'text': pos_tweets + neg_tweets,
    'labels': [1] * len(pos_tweets) + [0] * len(neg_tweets)
}

df = pd.DataFrame(data)


class Config:
    class ModelConfig:
        name = 'bert-base-uncased'
        batch_size = 32
        max_length = 128
        num_epochs = 5
        dataloader_workers = 2

    class GeneralConfig:
        validation_size = 0.1

    model = ModelConfig()
    general = GeneralConfig()


config = Config()


class BERTMAMBAModel(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=2):
        super(BERTMAMBAModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)

        for param in self.bert.parameters():
            param.requires_grad = True

        self.mamba_attention = nn.MultiheadAttention(self.bert.config.hidden_size, num_heads=8, dropout=0.1)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size)

        attn_output, _ = self.mamba_attention(hidden_state.permute(1, 0, 2), hidden_state.permute(1, 0, 2),
                                              hidden_state.permute(1, 0, 2))
        attn_output = attn_output.permute(1, 0, 2)  # Back to (batch_size, sequence_length, hidden_size)

        # Use the [CLS] token representation for classification
        cls_output = attn_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return logits
