from nltk.tokenize import TweetTokenizer
from emoji import demojize
import argparse
import wordsegment
import re
class Tokenizer():

    def __init__(self, reduce_len=False, post_process=False, segment_hashtags=False):
        self.post_process = post_process
        self.reduce_len = reduce_len
        self.segmentation = segment_hashtags
        wordsegment.load()
        self.tokenizer = TweetTokenizer(reduce_len=reduce_len)

    def tokenize_tweet(self, tweet:str):
        tokens = self.tokenizer.tokenize(tweet)
        post_tweet = " ".join([token for token in tokens])
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
        if self.segmentation:
            post_tweet = self.segment_hashtags(post_tweet)
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

    def segment_hashtags(self, tweet:str):
        hashtags = re.findall(r"#\w+", tweet)
        if len(hashtags) == 0:
            return tweet
        segmented_tweet = tweet
        for hashtag in hashtags:
            word = hashtag[1:]
            segmented = wordsegment.segment(word)
            replacement = " ".join(segmented)
            segmented_tweet = segmented_tweet.replace(hashtag, replacement)
        return segmented_tweet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduce_len", help="Reduces the length of the repeated characters to 3. Default: False")
    parser.add_argument("--post_process", help="To further process the tokens. Default: False")
    parser.add_argument("--segment_hashtags", help="Segments the hashtag into parts. Default False")
    args = parser.parse_args()

    tokenizer = Tokenizer(reduce_len=args.reduce_len, post_process=args.post_process, segment_hashtags=args.segment_hashtags)

    #tweet = "i'll eatttt  you like  😊 a piece of cake and you'll obey me!!!! i've done that already #ihatetaylorswift"
    #print(tokenizer.tokenize_tweet(tweet))