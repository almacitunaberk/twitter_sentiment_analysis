import os
import argparse
import numpy as np
import multiprocessing
from typing import List
import pandas as pd
from torchtext.vocab import GloVe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
import tqdm
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe

import sys
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
from tokenizer import Tokenizer

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, labels):
        self.tweets = tweets
        self.labels = labels
        self.glove = GloVe(name="twitter.27B", dim=100)

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        input_ids = self.glove.get_vecs_by_tokens(self.tweets[index]).view(-1)
        return input_ids, self.labels[index]

class BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 100
        self.input_size = 100
        self.num_layers = 1
        self.bidirectional = True
        self.num_directions = 1
        self.dropout1 = nn.Dropout(p=0.2)

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


def write_to_log(model_type:str,
                 log_path:str, log_filename:str,
                 mean_accuracy:float, std_accuracy:str,
                 glove_dim: int,
                 data_type:List[str],
                 model_args:dict=None):

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
    print("Loading GloVe embeddings")
    embeddings = {}
    with open(glove_path, "r") as f:
        for line in f:
            word, weights = line.split(maxsplit=1)
            weights = np.fromstring(weights, "f", sep=" ")
            embeddings[word] = weights
    print("Loaded GloVe embeddings")
    return embeddings

def tweet_embed(words, embeddings, glove_dim):
    unknown_indices = []
    mean = np.zeros(glove_dim)
    for i in range(len(words)):
        if words[i] in embeddings.keys():
            words[i] = embeddings[words[i]]
            mean += words[i]
        else:
            unknown_indices.append(i)
    if (len(words) - len(unknown_indices)) != 0:
        mean /= (len(words) - len(unknown_indices))
    for i in unknown_indices:
        words[i] = mean
    return np.array(words)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_batch = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_batch, labels

def get_length(tweet):
    return len(tweet)

def cross_validation(data, model_type:str, glove_dim:int, save_path:str):

    custom_tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)

    print("Transforming tweets")

    data["text"] = data["text"].apply(lambda x: custom_tokenizer.tokenize_tweet(x))
    data["length"] = data["text"].apply(get_length)
    data = data[data["length"] != 0]

    aggregated_acc = []

    train, val = train_test_split(data, test_size=0.2)

    train_tweets = train["text"].to_list()
    train_labels = train["labels"].to_list()

    val_tweets = val["text"].to_list()
    val_labels = val["labels"].to_list()

    print("Length of training samples {}".format(len(train_tweets)))
    print("Length of validation samples {}".format(len(val_tweets)))

    train_dataset = CustomDataset(train_tweets, torch.tensor(train_labels).float())
    val_dataset = CustomDataset(val_tweets, torch.tensor(val_labels).float())

    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)

    if model_type == "bilstm":
        bilstm = BiLSTM()
        bilstm.to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        val_losses = []
        val_accs = []
        train_losses = []
        train_accs = []
        for epoch in range(5):
            epoch_start_time = time.time()
            print("Epoch: {}".format(epoch+1))
            bilstm.train()
            train_loss = 0.0
            total = 0
            correct = 0
            for batch in train_loader:
                bilstm.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = torch.tensor(targets).float().unsqueeze(1).to(device)
                outputs = bilstm(inputs)
                loss = loss_fn(outputs, targets)
                train_loss += loss.item()
                predicted = (outputs >= 0.5).int()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)
            accuracy = 100 * correct / total
            train_accs.append(accuracy)


            bilstm.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = torch.tensor(targets).float().unsqueeze(1).to(device)
                    outputs = bilstm(inputs)
                    loss = loss_fn(outputs, targets)
                    val_loss += loss.item()
                    predicted = (outputs >= 0.5).int()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                val_loss /= len(val_loader)
                accuracy = 100 * correct / total
                val_accs.append(accuracy)
            scheduler.step()
            print("Epoch summary")
            print(f'Train Loss: {train_loss:7.2f}  Train Accuracy: {train_accs[-1]:6.3f}%')
            print(f'Validation Loss: {val_loss:7.2f}  Validation Accuracy: {val_accs[-1]:6.3f}%')
            print(f'Duration: {time.time() - epoch_start_time:.0f} seconds')
            print('')

        torch.save({
            "epoch": 5,
            "model_state_dict": bilstm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, f"{save_path}/bilstm_100_epochs_10_glove_dim_100_batch_size_32_lr_001.pt")
        val_accs = np.array(val_accs)
        return val_accs.mean(), val_accs.std()

if __name__ == "__main__":
    log_file = open("/home/ubuntu/bilstm_batched_logs.txt", "w")
    sys.stdout = log_file
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input path of the preprocessed csv file")
    parser.add_argument("--log_path", help="Path of the log folder")
    parser.add_argument("--log_filename", help="Name of the log file", default="logs")
    parser.add_argument("--save_path", help="Path for saving the model")
    #parser.add_argument("--glove_path", help="Path of the Glove embeddings", default="logs")
    args = parser.parse_args()
    if args.input_path is None or args.log_path is None or args.log_filename is None or args.save_path is None:
        print("Input path flag, log path, save_path or log filename flag cannot be none")
        exit()
    if not os.path.exists(f"{args.log_path}"):
        os.makedirs(f"{args.log_path}")
    if not os.path.exists(f"{args.save_path}"):
        os.makedirs(f"{args.save_path}")
    preprocessed_data_type = "BATCHED with 32"
    train_df = pd.read_csv(args.input_path)
    #train_df = train_df.sample(n=10000) # TODO: Comment this line when not testing
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
    mean_acc, std_acc = cross_validation(data=train_df, glove_dim=100, model_type="bilstm", save_path=args.save_path)
    write_to_log(glove_dim=100, model_type="bilstm",
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type=preprocessed_data_type, mean_accuracy=mean_acc, std_accuracy=std_acc)

    log_file.close()
    """
    for glove_dim in [100]:
        for model_type in ["bilstm"]:
            mean_accuracy = None
            std_accuracy = None
            mean_accuracy, std_accuracy = cross_validation(data=train_df,
                                                           glove_dim=glove_dim,
                                                            model_type=model_type,
                                                            save_path=args.save_path)
            write_to_log(glove_dim=glove_dim, model_type=model_type,
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type=preprocessed_data_type, mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)
    """

