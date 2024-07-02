import re
import nltk
import ssl
import os
import argparse

class Preprocessor():
    def __init__(self, input_path:str, output_dir:str, output_filename:str,
                 lowercase=True, no_url=False,
                 no_user=False, no_hashtag=False, no_numbers=False,
                 no_extra_space=True, no_stopwords=True, soft_lemmatize=True,
                 hard_lemmatize=False, stemming=False, slang_conversion=False):
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.lowercase = lowercase
        self.no_url = no_url
        self.no_user = no_user
        self.no_hashtag = no_hashtag
        self.no_numbers = no_numbers
        self.no_extra_space = no_extra_space
        self.no_stopwords = no_stopwords
        self.soft_lemmatize = soft_lemmatize
        self.hard_lemmatize = hard_lemmatize
        self.stemming = stemming
        self.slang_conversion = slang_conversion

    def process(self):
        tweets = []
        with open(self.input_path, "r", encoding="utf-8") as f:
            for line in f:
                tweets.append(line.rstrip())
        processed_tweets = []
        for tweet in tweets:
            if self.lowercase:
                tweet = self.lowercase_tweet(tweet=tweet)
            if self.no_url:
                tweet = self.remove_url(tweet=tweet)
            if self.no_user:
                tweet = self.remove_user(tweet=tweet)
            if self.no_hashtag:
                tweet = self.remove_hashtag(tweet=tweet)
            if self.no_numbers:
                tweet = self.remove_numbers(tweet=tweet)
            if self.no_extra_space:
                tweet = self.remove_extra_space(tweet=tweet)
            if self.no_stopwords:
                tweet = self.remove_stopwords(tweet=tweet)
            if self.soft_lemmatize:
                tweet = self.soft_lemmatize(tweet=tweet)
            if self.hard_lemmatize:
                tweet = self.hard_lemmatize(tweet=tweet)
            if self.stemming:
                tweet = self.stem_words(tweet=tweet)
            if self.slang_conversion:
                tweet = self.slang_process(tweet=tweet)
            processed_tweets.append(tweet)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(f"{self.output_dir}/{self.output_filename}.txt", "w") as f:
            for tweet in processed_tweets:
                tweet = tweet.strip()
                f.write(f"{tweet}\n")

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
            self.nltk_download_helper("stopwords")
            from nltk.corpus import stopwords
            stop_words = stopwords.words("english")
            return " ".join([word for word in tweet.split() if word not in stop_words])

    def soft_lemmatize(self, tweet:str):
        if self.soft_lemmatize:
            self.nltk_download_helper("wordnet")
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            return " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

    def hard_lemmatize(self, tweet:str):
        if self.hard_lemmatize:
            import spacy
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
            }
            return " ".join([slangs[word] if word in slangs else word for word in tweet.split()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Absolute path of the unprocessed input file")
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

    args = parser.parse_args()

    if args.input_path is None or args.output_dir is None or args.output_filename is None:
        raise ValueError("Input path, output directory and output filename are required arguments")

    preprocessor = Preprocessor(input_path=args.input_path, output_dir=args.output_dir,
                                output_filename=args.output_filename,
                                lowercase=args.lowercase, no_url=args.no_url, no_user=args.no_user,
                                no_hashtag=args.no_hashtag, no_numbers=args.no_numbers,
                                no_extra_space=args.no_extra_space,
                                no_stopwords=args.no_stopwords, soft_lemmatize=args.soft_lem,
                                hard_lemmatize=args.hard_lem, stemming=args.stemming,
                                slang_conversion=args.slang_conv)
    preprocessor.process()
