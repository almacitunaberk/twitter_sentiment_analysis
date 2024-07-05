import argparse
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import numpy as np

def cross_validation(tweets, labels,
                     embedding_model:str, model_type:str,
                     log_path:str, log_filename:str, data_type=[str],
                     embedding_args:dict=None, model_args:dict=None):

    if len(tweets) != len(labels):
        print("The length of the tweets and the labels do not match.")
        return

    if not embedding_model in ["bow", "tf-idf", "glove", "fasttext"]:
        print("The provided embedding model is not supported")
        return

    if not model_type in ["logistic", "random_forest", "lstm", "ridge", "gaussian", "multinomial"]:
        print("The provided model type is not supported")
        return

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aggregated_acc = []

    for i, (train_indices, test_indices) in enumerate(skf.split(tweets, labels)):
        training_tweets = tweets[train_indices]
        training_labels = labels[train_indices]

        val_tweets = tweets[test_indices]
        val_labels = labels[test_indices]

        accuracy = None
        model = None

        if embedding_model == "bow":
            vectorizer = CountVectorizer(max_features=embedding_args.get("max_features"))
            X_train = vectorizer.fit_transform(training_tweets)
            X_val = vectorizer.transform(val_tweets)
        elif embedding_model == "tf-idf":
            vectorizer = TfidfVectorizer(max_features=embedding_args.get("max_features"))
            X_train = vectorizer.fit_transform(training_tweets)
            X_val = vectorizer.transform(val_tweets)
        else:
            pass

        print("Training {model_type} with embedding type: {embedding_model}")
        if model_type == "logistic":
            model = LogisticRegression(n_jobs=model_args.get("n_jobs"), random_state=42, solver=model_args.get("solver"))
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
        elif model_type == "multinomial":
            model = MultinomialNB(alpha=model_args.get("alpha"))

        model.fit(X_train, training_labels)
        y_preds = model.predict(X_val)
        accuracy = accuracy_score(val_labels, y_preds)

        if accuracy == None:
            print("Accuracy gives None. Something went wrong!")
            return
        aggregated_acc.append(accuracy)
    mean_accuracy = np.array(aggregated_acc).mean()
    std_accuracy = np.array(aggregated_acc).std()
    print("{model_type} done. Accuracy: {mean_accuracy} Std: {std_accuracy}")
    print("Logging the results to the log file")
    log = f"{embedding_model}"
    for key in embedding_args:
        arg = embedding_args.get(key)
        log = f"{log} {key}:{arg}"
    log = f"{log} + {model_type}"
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
    """
    cross_validation(tweets=tweets, labels=labels,
                     embedding_model="bow", model_type="logistic",
                     model_args={"n_jobs":8, "solver":"saga"},
                     embedding_args={"max_features": 5000},
                     log_path=args.log_path, log_filename=args.log_filename,
                     data_type=preprocessed_data_type)

