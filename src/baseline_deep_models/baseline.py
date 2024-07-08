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
from tokenizer import Tokenizer as CustomTokenizer
from keras.preprocessing.text import Tokenizer

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

def cross_validation(tweets, labels, model_type:str, model_args:dict, glove_dim:int):
    processed_tweets = []
    lengths = []
    custom_tokenizer = CustomTokenizer(reduce_len=True, segment_hashtags=True)
    for tweet in tweets:
        tokenized_tweet = custom_tokenizer.tokenize_tweet(tweet=tweet)
        processed_tweets.append(" ".join(tokenized_tweet))
        lengths.append(len(tokenized_tweet))
    lengths = np.array(lengths)
    max_len = lengths.max()

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
    pos_tweets = np.array(train_df[train_df["labels"] == 1]["text"].values)[:1000]
    pos_labels = labels[:1000]
    neg_tweets = np.array(train_df[train_df["labels"] == 0]["text"].values)[:1000]
    neg_labels = [0 for i in range(1000)]
    tweets = np.concatenate([pos_tweets, neg_tweets])
    labels = np.concatenate([pos_labels, neg_labels])
    model_to_args = {
    }
    """
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
        for model_type in ["bilstm", "mlp", "cnn"]:
            mean_accuracy = None
            std_accuracy = None

            mean_accuracy, std_accuracy = cross_validation(tweets=tweets, labels=labels,
                                                           glove_dim=glove_dim,
                                                            model_type=model_type,
                                                            model_args=model_to_args.get(model_type))
            """
            write_to_log(glove_dim=glove_dim, model_type=model_type, model_args=model_to_args.get(model_type),
                         log_path=args.log_path, log_filename=args.log_filename,
                         data_type=preprocessed_data_type, mean_accuracy=mean_accuracy, std_accuracy=std_accuracy)
            """

