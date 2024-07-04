import re
import nltk
import ssl
import os
import argparse
import pandas as pd
import multiprocessing
import numpy as np
import spacy
from tqdm import tqdm

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")
nltk.download("stopwords")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class Preprocessor():
    def __init__(self, pos_input_path:str, neg_input_path, output_dir:str, output_filename:str,
                 lowercase=False, no_url=False,
                 no_user=False, no_hashtag=False, no_numbers=False,
                 no_extra_space=False, no_stopwords=False, soft_lem=False,
                 hard_lem=False, stemming=False, slang_conversion=False,
                 parallelize=False, num_parallels=1):
        self.pos_input_path = pos_input_path
        self.neg_input_path = neg_input_path
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.lowercase = lowercase
        self.no_url = no_url
        self.no_user = no_user
        self.no_hashtag = no_hashtag
        self.no_numbers = no_numbers
        self.no_extra_space = no_extra_space
        self.no_stopwords = no_stopwords
        self.soft_lem = soft_lem
        self.hard_lem = hard_lem
        self.stemming = stemming
        self.slang_conversion = slang_conversion
        self.parallelize = parallelize
        self.num_parallels = int(num_parallels) if num_parallels is not None else 1
        self._tweets = []
        self._labels = []

    def get_pos_and_neg_tweets(self):
        pos_tweets = []
        neg_tweets = []
        with open(self.pos_input_path, "r", encoding="utf-8") as f:
            for line in f:
                pos_tweets.append(line.rstrip())
        with open(self.neg_input_path, "r", encoding="utf-8") as f:
            for line in f:
                neg_tweets.append(line.rstrip())
        return pos_tweets, neg_tweets

    def split_dataset(self, data):
        return np.array_split(data, int(self.num_parallels))

    def _process_tweets(self, tweets, labels):
        processed_tweets = []
        for tweet in tweets:
            tweet = self.process_single_tweet(tweet=tweet)
            processed_tweets.append(tweet)
        if not os.path.exists(f"{self.output_dir}"):
            os.makedirs(self.output_dir)
        df = pd.DataFrame({"text": processed_tweets, "labels": labels})
        path = f"{self.output_dir}/{self.output_filename}.csv"
        df.to_csv(path, index=False)

    def _parallel_process_tweets(self, tweets, index, processed_list, total):
        tqdm_text = "#" + "{}".format(index).zfill(3)
        with tqdm(total=total, desc=tqdm_text, position=index+1) as pbar:
            for tweet in tweets:
                tweet = self.process_single_tweet(tweet=tweet)
                processed_list.append(tweet)
                pbar.update(1)
        return 1

    def process_single_tweet(self, tweet:str):
        processed_tweet = tweet
        if self.lowercase:
            processed_tweet = self.lowercase_tweet(tweet=processed_tweet)
        if self.no_url:
            processed_tweet = self.remove_url(tweet=processed_tweet)
        if self.no_user:
            processed_tweet = self.remove_user(tweet=processed_tweet)
        if self.no_hashtag:
            processed_tweet = self.remove_hashtag(tweet=processed_tweet)
        if self.no_numbers:
            processed_tweet = self.remove_numbers(tweet=processed_tweet)
        if self.no_extra_space:
            processed_tweet = self.remove_extra_space(tweet=processed_tweet)
        if self.no_stopwords:
            processed_tweet = self.remove_stopwords(tweet=processed_tweet)
        if self.soft_lem:
            processed_tweet = self.soft_lemmatize(tweet=processed_tweet)
        if self.hard_lem:
            processed_tweet = self.hard_lemmatize(tweet=processed_tweet)
        if self.stemming:
            processed_tweet = self.stem_words(tweet=processed_tweet)
        if self.slang_conversion:
            processed_tweet = self.slang_process(tweet=processed_tweet)
        return processed_tweet

    def process(self):
        pos_tweets, neg_tweets = self.get_pos_and_neg_tweets()
        if self.parallelize:
            splitted_pos_tweets = self.split_dataset(np.array(pos_tweets))
            splitted_neg_tweets = self.split_dataset(np.array(neg_tweets))
            for i in range(len(splitted_pos_tweets)):
                splitted_pos_tweets[i] = splitted_pos_tweets[i][:10]
                splitted_neg_tweets[i] = splitted_neg_tweets[i][:10]
            with multiprocessing.Manager() as manager:
                manager = multiprocessing.Manager()
                pos_processed_list = manager.list()
                neg_processed_list = manager.list()
                pool = multiprocessing.Pool(processes=self.num_parallels, initargs=(multiprocessing.RLock(),), initializer=tqdm.set_lock)
                jobs = [pool.apply_async(func=self._parallel_process_tweets, args=(splitted_pos_tweets[i], i, pos_processed_list, len(splitted_pos_tweets),)) for i in range(len(splitted_pos_tweets))]
                pool.close()
                results = []
                for job in tqdm(jobs):
                    results.append(job.get())
                print("\n" * (self.num_parallels+ 1))
                print(len(pos_processed_list))
                pool = multiprocessing.Pool(processes=self.num_parallels, initargs=(multiprocessing.RLock(),), initializer=tqdm.set_lock)
                jobs = [pool.apply_async(func=self._parallel_process_tweets, args=(splitted_neg_tweets[i], i, neg_processed_list, len(splitted_neg_tweets),)) for i in range(len(splitted_neg_tweets))]
                pool.close()
                results = []
                for job in tqdm(jobs):
                    results.append(job.get())
                print("\n" * (self.num_parallels+ 1))
                print(len(neg_processed_list))
                tweets = np.concatenate([pos_processed_list, neg_processed_list])
                pos_labels = [1 for i in range(len(pos_processed_list))]
                neg_labels = [0 for i in range(len(neg_processed_list))]
                labels = np.concatenate([pos_labels, neg_labels])
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                df = pd.DataFrame({"text": tweets, "labels": labels})
                path = f"{self.output_dir}/{self.output_filename}.csv"
                df.to_csv(path, index=False)
        else:
            tweets = np.concatenate([pos_tweets, neg_tweets])
            pos_labels = [1 for i in range(len(pos_tweets))]
            neg_labels = [0 for i in range(len(neg_tweets))]
            labels = np.concatenate([pos_labels, neg_labels])
            tweets = tweets[-10:]
            labels = labels[-10:]
            self._process_tweets(tweets, labels)

    def nltk_download_helper(self, source_name:str):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download(source_name)

    def lowercase_tweet(self, tweet:str):
        if self.lowercase:
            return tweet.lower()

    def remove_url(self, tweet: str):
        if self.no_url:
            return tweet.replace("<url>", "")

    def remove_user(self, tweet:str):
        if self.no_user:
            return tweet.replace("<user>", "")

    def remove_hashtag(self, tweet:str):
        if self.no_hashtag:
            return re.sub(r"#\w+", "", tweet)

    def remove_numbers(self, tweet:str):
        if self.no_numbers:
            return re.sub(r"\d+", "", tweet)

    def remove_extra_space(self, tweet:str):
        if self.no_extra_space:
            tweet = tweet.strip()
            return " ".join([re.sub(r"\s+", "", word) for word in tweet.split()])

    def remove_stopwords(self, tweet:str):
        if self.no_stopwords:
            stop_words = stopwords.words("english")
            return " ".join([word for word in tweet.split() if word not in stop_words])

    def soft_lemmatize(self, tweet:str):
        if self.soft_lemmatize:
            lemmatizer = WordNetLemmatizer()
            return " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

    def hard_lemmatize(self, tweet:str):
        if self.hard_lemmatize:
            spc = spacy.load("en_core_web_sm")
            doc = spc(tweet)
            tokens = [t for t in doc]
            return " ".join([token.lemma_ for token in tokens])

    def stem_words(self,tweet:str):
        if self.stemming:
            from nltk.stem.porter import PorterStemmer
            stemmer = PorterStemmer()
            return " ".join([stemmer.stem(word) for word in tweet.split()])

    def slang_process(self, tweet:str):
        if self.slang_conversion:
            slangs = {
                "imo": "in my opinion",
                "cyaa": "see you",
                "rn": "right now",
                "afaik": "as far as i know",
                "imma": "i am going to",
                "idk": "i do not know",
                "dont": "do not",
                "didnt": "did not",
                "ur": "your",
                "youre": "you are ",
                "won't": "will not",
                "gd": "good",
                "tht": "that",
                "&": "and",
                "@": "at",
                "pls": "please",
                "plz": "please",
                "..": "...",
            }
            return " ".join([slangs[word] if word in slangs else word for word in tweet.split()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_input_path", help="Absolute path of the unprocessed positive input file")
    parser.add_argument("--neg_input_path", help="Absolute path of the unprocessed negative input file")
    parser.add_argument("--output_dir", help="Absolute path of the output directory")
    parser.add_argument("--output_filename", help="Name of the output file")
    parser.add_argument("--lowercase", help="Default: True")
    parser.add_argument("--no_url", help="Default: False")
    parser.add_argument("--no_user", help="Default: False")
    parser.add_argument("--no_hashtag", help="Default: False")
    parser.add_argument("--no_numbers", help="Default: False")
    parser.add_argument("--no_extra_space", help="Default: True")
    parser.add_argument("--no_stopwords", help="Default: True")
    parser.add_argument("--soft_lem", help="Default: True")
    parser.add_argument("--hard_lem", help="Default: False")
    parser.add_argument("--stemming", help="Default: False")
    parser.add_argument("--slang_conv", help="Default: False")
    parser.add_argument("--parallelize", help="Given any value, it will evaluate to True always")
    parser.add_argument("--num_parallels", help="Default 1")

    args = parser.parse_args()

    if args.pos_input_path is None or args.neg_input_path is None or args.output_dir is None or args.output_filename is None:
        raise ValueError("Input path, output directory and output filename are required arguments!")

    preprocessor = Preprocessor(pos_input_path=args.pos_input_path, neg_input_path=args.neg_input_path,
                                output_dir=args.output_dir,output_filename=args.output_filename,
                                lowercase=args.lowercase, no_url=args.no_url, no_user=args.no_user,
                                no_hashtag=args.no_hashtag, no_numbers=args.no_numbers,
                                no_extra_space=args.no_extra_space,
                                no_stopwords=args.no_stopwords, soft_lem=args.soft_lem,
                                hard_lem=args.hard_lem, stemming=args.stemming,
                                slang_conversion=args.slang_conv, parallelize=args.parallelize,
                                num_parallels=args.num_parallels)
    preprocessor.process()
