import pickle
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import numpy as np
from torchtext.vocab import GloVe
import multiprocessing
import sys
filename = os.path.dirname(__file__)[:-1]
filename = "/".join(filename.split("/")[:-1])
sys.path.append(os.path.join(filename, 'preprocess'))
from tokenizer import Tokenizer
from typing import List
import logging
from tqdm import tqdm

#logger = logging.getLogger(__name__)

def write_to_log(embedding_model:str, embedding_args:dict,
                 model_type:str, model_args:dict,
                 log_path:str, log_filename:str,
                 accuracy:float,
                 data_type:List[str]):
    print("Logging the results to the log file")
    log = f"{embedding_model}"
    if embedding_args is not None:
        for key in embedding_args:
            arg = embedding_args.get(key)
            log = f"{log} {key}:{arg}"
        log = f"{log} + {model_type}"
    if model_args is not None:
        for key in model_args:
            arg = model_args.get(key)
            log = f"{log} {key}:{arg}"
    log = f"{log} \n"
    log = f"{log}\naccuracy: {accuracy}\n"
    with open(f"{log_path}/{log_filename}.txt", "a") as f:
        for word in data_type:
            f.write(f"{word} ")
        f.write("\n")
        f.write(log)
        f.write("\n-------------------------------------\n")
    print("Wrote to the log file")


def cross_validation(data, save_path:str,
                     embedding_model:str, model_type:str,
                     embedding_args:dict, model_args:dict,
                     log_path:str, log_filename:str):



    if not embedding_model in ["bow", "tf-idf", "glove"]:
        print("The provided embedding model is not supported")
        return

    if not model_type in ["logistic", "random_forest", "ridge", "gaussian", "bernoulli"]:
        print("The provided model type is not supported")
        return

    train, val = train_test_split(data, test_size=0.2)

    training_tweets = train["text"].values
    training_labels = train["labels"].values

    val_tweets = val["text"].values
    val_labels = val["labels"].values

    del train
    del val

    accuracy = None
    model = None

    if embedding_model == "bow":
        vectorizer = CountVectorizer(max_features=embedding_args.get("max_features"))
        X_train = vectorizer.fit_transform(training_tweets)
        X_val = vectorizer.transform(val_tweets)
        if model_type == "gaussian":
            X_train = np.asarray(X_train.todense())
            X_val = np.asarray(X_val.todense())
    elif embedding_model == "tf-idf":
        vectorizer = TfidfVectorizer(max_features=embedding_args.get("max_features"))
        X_train = vectorizer.fit_transform(training_tweets)
        X_val = vectorizer.transform(val_tweets)
        if model_type == "gaussian":
            X_train = np.asarray(X_train.todense())
            X_val = np.asarray(X_val.todense())
    elif embedding_model == "glove":
        glove = GloVe(name="twitter.27B", cache="/home/ubuntu/twitter_sentiment_analysis/.vector_cache", dim=embedding_args.get("dimension"))
        tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True)
        X_train = np.array([np.mean(glove.get_vecs_by_tokens(tokenizer.tokenize_tweet(tweet=tweet), lower_case_backup=True).numpy(), axis=0) for tweet in training_tweets if len(tweet) != 0], dtype="float64")
        X_val = np.array([np.mean(glove.get_vecs_by_tokens(tokenizer.tokenize_tweet(tweet=tweet), lower_case_backup=True).numpy(), axis=0) for tweet in val_tweets if len(tweet) != 0], dtype="float64")

    if embedding_model in ["bow", "tf-idf"]:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    print(f"Training {model_type} with embedding type: {embedding_model}")
    if model_type == "logistic":
        model = LogisticRegression(n_jobs=model_args.get("n_jobs"),
                                    random_state=42,
                                    solver=model_args.get("solver"),
                                    C=model_args.get("C"), max_iter=model_args.get("max_iter"))
    elif model_type == "ridge":
        model = Ridge(alpha=model_args.get("alpha"), random_state=42, max_iter=model_args.get("max_iter"), solver=model_args.get("solver"))
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_jobs=model_args.get("n_jobs"),
                                    random_state=42,
                                    n_estimators=model_args.get("n_estimators"),
                                    max_depth=model_args.get("max_depth"),
                                    max_features=model_args.get("max_features"))
    elif model_type == "gaussian":
        model = GaussianNB()
    elif model_type == "bernoulli":
        model = BernoulliNB(alpha=model_args.get("alpha"))

    model.fit(X_train, training_labels)
    del X_train
    del training_tweets
    del training_labels
    if model_type == "ridge":
        y_preds = (model.predict(X_val) > 0.5).astype(np.int64)
    else:
        y_preds = model.predict(X_val)

    accuracy = accuracy_score(val_labels, y_preds)
    del X_val
    del val_tweets
    del val_labels

    if accuracy == None:
        print("Accuracy gives None. Something went wrong!")
        return

    print(f"{model_type} done. Accuracy: {accuracy}")
    model_save_name = f"{embedding_model}"
    if embedding_args is not None:
        for key in embedding_args:
            arg = embedding_args.get(key)
            model_save_name = f"{model_save_name}_{key}_{arg}"
        model_save_name = f"{model_save_name}_{model_type}"
    if model_args is not None:
        for key in model_args:
            arg = model_args.get(key)
            model_save_name = f"{model_save_name}_{key}_{arg}"
    model_save_name = f"{model_save_name}.pkl"
    with open(f"{save_path}/{model_save_name}", "wb") as f:
        pickle.dump(model, f, protocol=5)
    del model
    write_to_log(embedding_model=embedding_model, embedding_args=embedding_args,
                            model_type=model_type, model_args=model_args, log_path=log_path,
                            log_filename=log_filename, data_type="preprocessed",
                            accuracy=accuracy)
    print(f"{model_type} with {embedding_model} is trained")

def get_length(tweet):
    return len(tweet)

if __name__ == "__main__":
    logging.basicConfig(filename="non_deep_model_logs.txt",
                        filemode="a")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input path of the preprocessed csv file")
    parser.add_argument("--log_path", help="Path of the log folder")
    parser.add_argument("--log_filename", help="Name of the log file", default="logs")
    parser.add_argument("--save_path")
    args = parser.parse_args()
    if args.input_path is None or args.log_path is None or args.log_filename is None or args.save_path is None:
        print("Input path flag, save path, log path or log filename flag cannot be none")
        exit()
    if not os.path.exists(f"{args.log_path}"):
        os.makedirs(f"{args.log_path}")
    if not os.path.exists(f"{args.save_path}"):
        os.makedirs(f"{args.save_path}")
    preprocessed_data_type = args.input_path.split("/")[-1][:-4].split("_")[1:]
    train_df = pd.read_csv(args.input_path)
    #train_df = train_df.sample(10000)
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
    """
    model_to_args = {
        "logistic": {
            "n_jobs": multiprocessing.cpu_count(),
            "solver": "saga",
            "C": 1e5,
            "max_iter": 100,
        },
        "ridge": {
            "alpha": 0.1,
            "max_iter": 1000,
            "solver": "auto",
        },
        "random_forest_glove": {
            "n_estimators": 100,
            "max_depth": 10,
            "max_features": 50,
            "n_jobs": 1,
        },
        "random_forest_others": {
            "n_jobs": multiprocessing.cpu_count(),
            "n_estimators": 100,
            "max_depth": 50,
            "max_features": 50,
        },
        "gaussian": None,
        "bernoulli": {
            "alpha": 0.01,
        }
    }
    embedding_to_args = {
        "bow": {
            "max_features": 5000,
        },
        "tf-idf": {
            "max_features": 5000,
        },
        "glove": {
            "dimension": 100,
        }
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
    init_args = []
    for embedding_model in ["bow", "tf-idf"]:
            for model_type in ["ridge", "bernoulli"]:
                model_to_args_key = model_type

                if model_type == "random_forest" and embedding_model == "glove":
                    model_to_args_key = "random_forest_glove"
                elif model_type == "random_forest" and embedding_model != "glove":
                    model_to_args_key = "random_forest_others"

                init_args.append((train_df,
                                  args.save_path,
                                  embedding_model,
                                  model_type,
                                  embedding_to_args.get(embedding_model),
                                  model_to_args.get(model_to_args_key),
                                  args.log_path,
                                  args.log_filename,
                                  ))

    """
    with multiprocessing.Manager() as manager:
            pool = multiprocessing.Pool(processes=16)
            jobs = [pool.apply_async(func=cross_validation, args=init_args[i]) for i in range(len(init_args))]
            pool.close()
            results = []
            with tqdm(total=len(init_args)) as pbar:
                for job in jobs:
                    result = job.get()
                    results.append(result)
                    pbar.update(1)
    """
    """
                accuracy = cross_validation(data=train_df,
                                            embedding_model=embedding_model,
                                            model_type=model_type,
                                            embedding_args=embedding_to_args.get(embedding_model),
                                            model_args=model_to_args.get(model_to_args_key),
                                            save_path=args.save_path)

                write_to_log(embedding_model=embedding_model, embedding_args=embedding_to_args.get(embedding_model),
                            model_type=model_type, model_args=model_to_args.get(model_to_args_key), log_path=args.log_path,
                            log_filename=args.log_filename, data_type=preprocessed_data_type,
                            accuracy=accuracy)
    """
    train_df["length"] = train_df["text"].apply(get_length)
    train_df = train_df[train_df["length"] != 0]


    for embedding_model in ["glove"]:
        for model_type in ["logistic", "random_forest"]:

            model_to_args_key = model_type

            if model_type == "random_forest" and embedding_model == "glove":
                model_to_args_key = "random_forest_glove"
            elif model_type == "random_forest" and embedding_model != "glove":
                model_to_args_key = "random_forest_others"

            cross_validation(train_df,
                        args.save_path,
                        embedding_model,
                        model_type,
                        embedding_to_args.get(embedding_model),
                        model_to_args.get(model_to_args_key),
                        args.log_path,
                        args.log_filename)

