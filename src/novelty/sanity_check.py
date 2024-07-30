import pandas as pd
import numpy as np
from ast import literal_eval
import torch
df = pd.read_csv("/home/ubuntu/preds/preds.csv")
def makeArray(rawdata):
        string = literal_eval(rawdata)
        return np.array(string)

# Applying the function row-wise, there could be a more efficient way
df['predictions'] = df['predictions'].apply(lambda x: makeArray(x))

predictions = np.array(df["predictions"].values, dtype=float)
tensors = torch.from_numpy(predictions)
print(tensors.size())
print(type(predictions))
print(predictions.shape)