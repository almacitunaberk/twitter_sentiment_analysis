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

def cross_validation(tweets, labels, model_type:str, glove_dim:int):
    processed_tweets = []
    lengths = []
    custom_tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True, post_process=True)
    for tweet in tweets:
        processed_tweet = custom_tokenizer.tokenize_tweet(tweet=tweet)
        lengths.append(len(processed_tweet))
    lengths = np.array(lengths)
    processed_tweets = np.array(processed_tweets)
    max_len = lengths.max()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_tweets)
    vocab_size = len(tokenizer.word_index)+1
    encoded_tweets = tokenizer.texts_to_sequences(processed_tweets)
    padded_tweets = pad_sequences(encoded_tweets, maxlen=max_len, padding="post")

    embed_matrix = np.zeros((vocab_size, glove_dim))
    glove = GloVe(name="twitter.27B", dim=glove_dim)
    for word, i in tokenizer.word_index.items():
        embedding_vec = glove.get_vecs_by_tokens(word)
        if embedding_vec is not None:
            embed_matrix[i] = embedding_vec

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aggregated_acc = []

    for i, (train_indices, test_indices) in enumerate(skf.split(padded_tweets, labels)):
        training_tweets = padded_tweets[train_indices]
        training_labels = labels[train_indices]

        val_tweets = padded_tweets[test_indices]
        val_labels = labels[test_indices]

        accuracy = None
        model = None

        if model_type == "bilstm":
            model = Sequential()
            model.add(Embedding(vocab_size, glove_dim, input_length=max_len, weights=[embed_matrix], trainable=False))
            model.add(Bidirectional(LSTM(20, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Bidirectional(LSTM(20, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Bidirectional(LSTM(20)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(1, activation="sigmoid"))
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.fit(training_tweets, training_labels, epochs=10, batch_size=64)
            #y_preds = model.predict(val_tweets)
            #accuracy = accuracy_score(val_labels, y_preds)
            # accuracy = keras.losses.binary_crossentropy(val_labels, y_preds)
            loss_and_metrics = model.evaluate(val_tweets, val_labels)
            print(f"Loss: {loss_and_metrics[0]}")
            print(f"Accuracy: {loss_and_metrics[1]}")
            accuracy = loss_and_metrics[1]
        aggregated_acc.append(accuracy)
    print(aggregated_acc)
    mean_accuracy = np.array(aggregated_acc).mean()
    std_accuracy = np.array(aggregated_acc).std()
    print(f"Accuracy: {mean_accuracy}, Std: {std_accuracy}")
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
    tweets = np.array(train_df["text"].values)
    labels = np.array(train_df["labels"].values)
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
            mean_accuracy, std_accuracy = cross_validation(tweets=tweets, labels=labels,
                                                           glove_dim=glove_dim,
                                                            model_type=model_type)
            write_to_log(glove_dim=glove_dim, model_type=model_type, model_args=model_to_args.get(model_type),
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type=preprocessed_data_type, mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)

