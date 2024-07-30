import pandas as pd
import numpy as np
from ast import literal_eval
import torch

tokens = np.load("/home/ubuntu/twitter_sentiment_analysis/src/cls_tokens/vinai-base.npy", allow_pickle=True)
print(len(tokens))
print(len(tokens[0]))
print(tokens.shape)