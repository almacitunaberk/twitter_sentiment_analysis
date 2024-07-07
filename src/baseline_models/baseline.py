import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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
print(sys.path[-1])
from tokenizer import Tokenizer
"""
def load_glove_embeddings():
    embedding_dict = {}
    with open("../glove.twitter.27B.100d.txt", encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

# Function to convert tweets to GloVe embeddings
def tweet_to_glove_embedding(tweet, embedding_dict):
    words = tweet.split()
    embeddings = [embedding_dict[word] for word in words if word in embedding_dict]
    if len(embeddings) == 0:
        return np.zeros(len(embedding_dict[next(iter(embedding_dict))]))
    return np.mean(embeddings, axis=0)
"""

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

    if not model_type in ["logistic", "random_forest", "lstm", "ridge", "gaussian", "bernoulli"]:
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
            glove = GloVe(name="twitter.27B", dim=embedding_args.get("dimension"))
            tokenizer = Tokenizer(reduce_len=True, segment_hashtags=True)
            X_train = np.array([np.mean(glove.get_vecs_by_tokens(tokenizer.tokenize_tweet(tweet=tweet), lower_case_backup=True).numpy(), axis=0) for tweet in training_tweets], dtype="float64")
            X_val = np.array([np.mean(glove.get_vecs_by_tokens(tokenizer.tokenize_tweet(tweet=tweet), lower_case_backup=True).numpy(), axis=0) for tweet in val_tweets], dtype="float64")

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
            if embedding_model == "bow":
                X_train
            model = GaussianNB()
        elif model_type == "bernoulli":
            model = BernoulliNB(alpha=model_args.get("alpha"))

        model.fit(X_train, training_labels)
        if model_type == "ridge":
            y_preds = (model.predict(X_val) > 0.5).astype(np.int64)
        else:
            y_preds = model.predict(X_val)
        accuracy = accuracy_score(val_labels, y_preds)

        if accuracy == None:
            print("Accuracy gives None. Something went wrong!")
            return
        aggregated_acc.append(accuracy)

    mean_accuracy = np.array(aggregated_acc).mean()
    std_accuracy = np.array(aggregated_acc).std()
    print(f"{model_type} done. Accuracy: {mean_accuracy} Std: {std_accuracy}")
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
    pos_tweets = np.array(train_df[train_df["labels"] == 1]["text"].values)[:1000]
    pos_labels = labels[:1000]
    neg_tweets = np.array(train_df[train_df["labels"] == 0]["text"].values)[:1000]
    neg_labels = [0 for i in range(1000)]
    tweets = np.concatenate([pos_tweets, neg_tweets])
    labels = np.concatenate([pos_labels, neg_labels])
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
            "n_jobs": multiprocessing.cpu_count(),
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
            "dimension": 200,
        }
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
    for embedding_model in ["bow", "tf-idf", "glove"]:
        for model_type in ["logistic", "ridge", "random_forest", "gaussian", "bernoulli"]:
            if model_type == "random_forest":
                if embedding_model == "glove":
                    cross_validation(tweets=tweets, labels=labels,
                                embedding_model=embedding_model,
                                model_type=model_type,
                                embedding_args=embedding_to_args.get(embedding_model),
                                model_args=model_to_args.get("random_forest_glove"),
                                log_path=args.log_path,
                                log_filename=args.log_filename,
                                data_type=preprocessed_data_type)
                else:
                    cross_validation(tweets=tweets, labels=labels,
                                embedding_model=embedding_model,
                                model_type=model_type,
                                embedding_args=embedding_to_args.get(embedding_model),
                                model_args=model_to_args.get("random_forest_others"),
                                log_path=args.log_path,
                                log_filename=args.log_filename,
                                data_type=preprocessed_data_type)
            else:
                cross_validation(tweets=tweets, labels=labels,
                                embedding_model=embedding_model,
                                model_type=model_type,
                                embedding_args=embedding_to_args.get(embedding_model),
                                model_args=model_to_args.get(model_type),
                                log_path=args.log_path,
                                log_filename=args.log_filename,
                                data_type=preprocessed_data_type)
        """