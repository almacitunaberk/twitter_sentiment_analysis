import os
import argparse
import numpy as np
import multiprocessing
from typing import List
import pandas as pd

import sys
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))

from tokenizer import Tokenizer
from torchtext.vocab import GloVe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

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

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size*self.num_directions, 1)

    def forward(self, tweet):
        lstm_out, _ = self.lstm(tweet.view(len(tweet), 1, -1))
        x = self.dropout1(lstm_out.view(len(tweet), -1))
        output = self.fc(x)
        pred = torch.sigmoid(output[-1])
        return pred


def write_to_log(model_type:str, model_args:dict,
                 log_path:str, log_filename:str,
                 mean_accuracy:float, std_accuracy:str,
                 glove_dim: int,
                 data_type:List[str]):

    print("Logging the results to the log file")
    log = f"glove {glove_dim} + {model_type}"
    if model_args is not None:
        for key in model_args:
            arg = model_args.get(key)
            log = f"{log} {key}:{arg}"
    log = f"{log} \n"
    log = f"{log}\naccuracy: {mean_accuracy} std: {std_accuracy}\n"
    with open(f"{log_path}/{log_filename}.txt", "a") as f:
        for word in data_type:
            f.write(f"{word} ")
        f.write("\n")
        f.write(log)
        f.write("\n-------------------------------------\n")
    print("Wrote to the log file")

def load_glove_embeddings(glove_path: str):
    embeddings = {}
    with open(glove_path, "r") as f:
        for line in f:
            word, weights = line.split(maxsplit=1)
            weights = np.fromstring(weights, "f", sep=" ")
            embeddings[word] = weights
    return embeddings

def tweet_embed(words, embeddings):
    unknown_indices = []
    mean = np.zeros(len(embeddings["king"]))
    for i in range(len(words)):
        if words[i] in embeddings.keys():
            words[i] = embeddings[i]
            mean += words[i]
        else:
            unknown_indices.append(i)
    if (len(words) - len(unknown_indices)) != 0:
        mean /= (len(words) - len(unknown_indices))
    for i in unknown_indices:
        words[i] = mean
    return np.array(words)

def cross_validation(data, model_type:str, glove_dim:int):
    custom_tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)

    data["text"] = data["text"].apply(lambda x: custom_tokenizer.tokenize_tweet(x))
    data["text"] = data["text"].apply(lambda x: tweet_embed(x))

    mask_zero_length = (data["text"].str.len() != 0)
    data = data.iloc[mask_zero_length]

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    aggregated_acc = []

    tweets = data["text"].values
    labels = data["text"].values

    aggregated_acc = []

    for i, (train_indices, test_indices) in enumerate(skf.split(tweets, labels)):
        training_tweets = tweets[train_indices]
        training_labels = labels[train_indices]

        val_tweets = tweets[test_indices]
        val_labels = labels[test_indices]

        if model_type == "bilstm":
            bilstm = BiLSTM(glove_dim, 0.2)
            loss_func = nn.BCELoss()
            optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            for epoch in range(4):
                print("Epoch: {}".format(epoch+1))
                train_loss = 0
                correct = 0
                bilstm.train()
                for i in range(len(training_tweets)):
                    bilstm.zero_grad()
                    tweet = torch.FloatTensor(training_tweets[i])
                    label = torch.FloatTensor(np.array([training_labels[i]]))
                    if torch.cuda.is_available():
                        tweet = tweet.cuda()
                        label = label.cuda()
                    pred = bilstm(tweet)
                    loss = loss_func(pred, label)
                    lambda_param = torch.tensor(0.001)
                    l2_reg = torch.tensor(0.)

                    if torch.cuda.is_available():
                        lambda_param = lambda_param.cuda()
                        l2_reg = l2_reg.cuda()
                    for param in bilstm.parameters():
                        if torch.cuda.is_available():
                            l2_reg += torch.norm(param).cuda()
                        else:
                            l2_reg += torch.norm(param)
                    loss += lambda_param * l2_reg

                    pred = pred.item()
                    if pred > 0.5:
                        pred = 1
                    else:
                        pred = 0
                    if pred == int(label.item()):
                        correct += 1
                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (i+1)%1000 == 0:
                        print("Processed {} tweets out of {}".format(i+1, len(training_tweets)))
                train_losses.append(train_loss/len(training_tweets))
                train_accs.append(correct / len(training_tweets))

                val_loss = 0
                correct = 0
                bilstm.eval()
                with torch.no_grad():
                    for i in range(len(val_tweets)):
                        tweet = torch.FloatTensor(val_tweets[i])
                        label = torch.FloatTensor(np.array([val_labels[i]]))

                        if torch.cuda.is_available():
                            tweet = tweet.cuda()
                            label = label.cuda()
                        pred = bilstm(tweet)
                        loss = loss_func(pred, label)
                        val_loss += loss.item()
                        pred = pred.item()
                        if pred > 0.5:
                            pred = 1
                        else:
                            pred = 0
                        if pred == int(label.item()):
                            correct += 1
                val_losses.append(val_loss/len(val_tweets))
                val_accs.append(correct/len(val_labels))
                print("Epoch summary")
                print(f'Train Loss: {train_losses[-1]:7.2f}  Train Accuracy: {train_accuracies[-1]*100:6.3f}%')
                print(f'Validation Loss: {val_losses[-1]:7.2f}  Validation Accuracy: {val_accuracies[-1]*100:6.3f}%')
                print(f'Duration: {time.time() - epoch_start_time:.0f} seconds')
                print('')

                scheduler.step()
            aggregated_acc.append(val_accs[-1])
            mean_accuracy = np.array(aggregated_acc).mean()
            std_accuracy = np.array(aggregated_acc).std()
            return mean_accuracy, std_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input path of the preprocessed csv file")
    parser.add_argument("--log_path", help="Path of the log folder")
    parser.add_argument("--log_filename", help="Name of the log file", default="logs")
    args = parser.parse_args()
    if args.input_path is None or args.log_path is None or args.log_filename is None:
        print("Input path flag, log path or log filename flag cannot be none")
        exit()
    if not os.path.exists(f"{args.log_path}"):
        os.makedirs(f"{args.log_path}")
    preprocessed_data_type = args.input_path.split("/")[-1][:-4].split("_")[1:]
    train_df = pd.read_csv(args.input_path)
    train_df = train_df.dropna()
    """
    pos_tweets = np.array(train_df[train_df["labels"] == 1]["text"].values)[:1000]
    pos_labels = labels[:1000]
    neg_tweets = np.array(train_df[train_df["labels"] == 0]["text"].values)[:1000]
    neg_labels = [0 for i in range(1000)]
    tweets = np.concatenate([pos_tweets, neg_tweets])
    labels = np.concatenate([pos_labels, neg_labels])
    model_to_args = {
    }
    cross_validation(tweets=tweets, labels=labels,
                                embedding_model="glove",
                                model_type="ridge",
                                embedding_args=embedding_to_args.get("glove"),
                                model_args=model_to_args.get("ridge"),
                                log_path=args.log_path,
                                log_filename=args.log_filename,
                                data_type=preprocessed_data_type)
    """
    for glove_dim in [100, 200]:
        for model_type in ["bilstm"]:
            mean_accuracy = None
            std_accuracy = None
            mean_accuracy, std_accuracy = cross_validation(data=train_df,
                                                           glove_dim=glove_dim,
                                                            model_type=model_type)
            write_to_log(glove_dim=glove_dim, model_type=model_type, model_args=model_to_args.get(model_type),
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type=preprocessed_data_type, mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)

