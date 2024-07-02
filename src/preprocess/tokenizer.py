from nltk.tokenize import TweetTokenizer
from emoji import demojize
import argparse

class Tokenizer():

    def __init__(self, reduce_len=False, post_process=False):
        self.post_process = post_process
        self.tokenizer = TweetTokenizer(reduce_len=reduce_len)

    def tokenize_tweet(self, tweet:str):
        tokens = self.tokenizer.tokenize(tweet)
        post_tweet = " ".join([token in tokens if not self.post_process else self.process(token) for token in tokens])
        if self.post_process:
            post_tweet = (
                post_tweet.replace("cannot ", "can not ")
                .replace("n't ", " n't ")
                .replace("n 't ", "n't ")
                .replace("ca n't", "can't")
                .replace("ai n't", "ain't")
                .replace("'m ", " am ")
                .replace("'re ", " are ")
                .replace("'s ", " 's ")
                .replace("'ll ", " will ")
                .replace("'d ", " 'd ")
                .replace("'ve ", " have ")
                )
        return post_tweet.split()

    def process(self, token:str):
        if len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduce_len", help="Reduces the length of the repeated characters to 3. Default: False")
    parser.add_argument("--post_process", help="To further process the tokens. Default: False")

    args = parser.parse_args()

    tokenizer = Tokenizer(reduce_len=args.reduce_len, post_process=args.post_process)

    tweet = "i'll eatttt  you like  😊 a piece of cake and you'll obey me!!!! i've done that already"
    print(tokenizer.tokenize_tweet(tweet))