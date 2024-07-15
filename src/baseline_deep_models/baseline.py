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
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchtext import datasets
from torchtext import data
import torchtext

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

#device = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

class BiLSTM(nn.Module):
    def __init__(self, embed_dim, drop_prob):
        super().__init__()
        self.hidden_size = 50
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

class CustomDataset(torchtext.data.Dataset):
    def __init__(self, data, TEXT, LABEL):
        fields = [("tweet", TEXT), ("label", LABEL), (None, None)]
        dataset = []
        for i in range(data.shape[0]):
            tweet = data.text[i]
            label = data.labels[i]
            dataset.append(data.Example.fromlist([text, label], fields))

    @classmethod
    def splits(cls, text_field, label_field, root='data',
               train='train', test='test', **kwargs):
        return super().splits(
            root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)


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

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def cross_validation(df, model_type:str, glove_dim:int, glove_path:str):

    custom_tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)

    embeddings = load_glove_embeddings(glove_path)

    print("Transforming tweets")

    df["text"] = df["text"].apply(lambda x: custom_tokenizer.tokenize_tweet(x))
    #data[data["text"].map(len) >= 1]
    df["text"] = df["text"].apply(lambda x: tweet_embed(x, embeddings, glove_dim))

    train, valid = train_test_split(df, test_size=0.2, random_state=42)

    text_field = data.Field()
    label_field = data.LabelField(dtype=torch.float)
    fields = [('text',text_field),('label',label_field),(None, None)]

    train = CustomDataset(train,text_field,label_field)
    valid = CustomDataset(valid,text_field,label_field)

    print("Number of training samples: {}".format(len(train)))
    print("Number of validation samples: {}".format(len(valid)))

    BATCH_SIZE = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test), batch_size=BATCH_SIZE,device=device)
    train_iterator = data.BucketIterator(train_data , batch_size=BATCH_SIZE,device=device)
    valid_iterator = data.BucketIterator(valid_data, batch_size=BATCH_SIZE,device=device)
    test_iterator = data.BucketIterator(test, batch_size=BATCH_SIZE,device=device)

    print("Length of train iterator: {}".format(len(train_iterator)))

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    aggregated_acc = []

    tweets = data["text"].values
    labels = data["labels"].values

    """
    aggregated_acc = []

    for i, (train_indices, test_indices) in enumerate(skf.split(tweets, labels)):
        training_tweets = tweets[train_indices]
        training_labels = labels[train_indices]

        val_tweets = tweets[test_indices]
        val_labels = labels[test_indices]

        non_zero_indices = []

        for i, tweet in enumerate(training_tweets):
            if len(tweet) != 0:
                non_zero_indices.append(i)

        training_tweets = np.asarray(training_tweets[non_zero_indices])
        training_labels = np.asarray(training_labels[non_zero_indices])

        print("Length of training tweets {}".format(training_tweets.size))

        non_zero_indices = []

        for i, tweet in enumerate(val_tweets):
            if len(tweet) != 0:
                non_zero_indices.append(i)

        val_tweets = np.asarray(val_tweets[non_zero_indices])
        val_labels = np.asarray(val_labels[non_zero_indices])

        print("Length of validation tweets {}".format(val_tweets.size))

        if model_type == "bilstm":
            bilstm = BiLSTM(glove_dim, 0.2)
            bilstm.cuda()
            loss_func = nn.BCELoss()
            optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            for epoch in range(2):
                epoch_start_time = time.time()
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
                print(f'Train Loss: {train_losses[-1]:7.2f}  Train Accuracy: {train_accs[-1]*100:6.3f}%')
                print(f'Validation Loss: {val_losses[-1]:7.2f}  Validation Accuracy: {val_accs[-1]*100:6.3f}%')
                print(f'Duration: {time.time() - epoch_start_time:.0f} seconds')
                print('')

                scheduler.step()
            aggregated_acc.append(val_accs[-1])
            mean_accuracy = np.array(aggregated_acc).mean()
            std_accuracy = np.array(aggregated_acc).std()
            return mean_accuracy, std_accuracy
    """
    if model_type == "bilstm":
        epoch_loss = 0
        bilstm = BiLSTM(glove_dim, 0.2)
        bilstm.cuda()
        loss_func = nn.BCELoss()
        optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        counter=1
        acc=[]
        for epoch in range(2):
            bilstm.train()
            for batch in train_iterator:
                optimizer.zero_grad()
                predictions = bilstm(batch.text).squeeze(1)
                loss = loss_func(predictions, batch.label)
                #rl.append(loss)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                counter=counter+1

                if(counter%2010==0):
                    print('Epoch: [{}/{}] | Step: [{}/{}] | Loss: {} |'.format(epochs+1, num_epochs, int((counter)/2010),10 , round(epoch_loss/2010,3)))
                    #counter=0
                    epoch_loss=0
            epoch_acc = 0
            bilstm.eval()
            for batch in valid_iterator:
                predictions = bilstm(batch.text).squeeze(1)
                accuracy = binary_accuracy(predictions, batch.label)
                epoch_acc += accuracy.item()
            acc.append(valid_acc)
            print(f'| Epoch: {epochs+1:02} | Val. Acc: {valid_acc*100:.2f}% | ---%Saving the model%---')


    #return epoch_loss / len(iterator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input path of the preprocessed csv file")
    parser.add_argument("--log_path", help="Path of the log folder")
    parser.add_argument("--log_filename", help="Name of the log file", default="logs")
    parser.add_argument("--glove_path", help="Path of the Glove embeddings", default="logs")
    args = parser.parse_args()
    if args.input_path is None or args.log_path is None or args.log_filename is None or args.glove_path is None:
        print("Input path flag, log path, glove_path or log filename flag cannot be none")
        exit()
    if not os.path.exists(f"{args.log_path}"):
        os.makedirs(f"{args.log_path}")
    preprocessed_data_type = args.input_path.split("/")[-1][:-4].split("_")[1:]
    train_df = pd.read_csv(args.input_path)
    train_df = train_df.sample(n=100000) # TODO: Comment this line when not testing
    train_df = train_df.dropna()
    print("Started with data length: {}".format(train_df.shape[0]))
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

    for glove_dim in [100]:
        for model_type in ["bilstm"]:
            mean_accuracy = None
            std_accuracy = None
            mean_accuracy, std_accuracy = cross_validation(data=train_df,
                                                           glove_dim=glove_dim,
                                                            model_type=model_type,
                                                            glove_path=args.glove_path)
            write_to_log(glove_dim=glove_dim, model_type=model_type,
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type="processed", mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)

