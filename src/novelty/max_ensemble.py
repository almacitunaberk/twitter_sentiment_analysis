from ast import literal_eval
import pandas as pd
import numpy as np
from tqdm import tqdm
df = pd.read_csv("/home/ubuntu/preds/preds_test.csv")

def makeArray(rawdata):
        string = literal_eval(rawdata)
        return np.array(string)

df['predictions'] = df['predictions'].apply(lambda x: makeArray(x))
preds = df["predictions"].values
final_preds = []
for i in range(len(preds)):
    final_preds.append(np.max(preds[i]))
    final_preds[i] = 1 if final_preds[i] >= 0.5 else -1

pred_df = pd.DataFrame()
pred_df["Id"] = np.arange(1, len(final_preds)+1)
pred_df["Prediction"] = np.array(final_preds).ravel()
pred_df["Prediction"] = pred_df["Prediction"].replace(0, -1)
pred_df.to_csv(f"/home/ubuntu/submission/max_ensemble.csv", index=False)
