from nltk.tokenize import TweetTokenizer
from emoji import demojize
import argparse
import wordsegment
import re

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}

CONTRACTIONS_PATTERN = re.compile(
    "({})".format("|".join(CONTRACTION_MAP.keys())),
    flags=re.IGNORECASE | re.DOTALL,
)

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
            def expand_match(contraction):
                match = contraction.group(0)
                first_char = match[0]
                expand_contraction = (
                    CONTRACTION_MAP.get(match)
                    if CONTRACTION_MAP.get(match)
                    else CONTRACTION_MAP.get(match.lower())
                )
                return first_char + expand_contraction[1:]
            post_tweet = CONTRACTIONS_PATTERN.sub(expand_match, post_tweet)
            post_tweet = re.sub("'", " ", post_tweet)
        if self.segmentation:
            post_tweet = self.segment_hashtags(post_tweet)
        return post_tweet.split()

    """
    def expand_contractions(self, tweet: str):

        def expand_match(contraction):
            match = contraction.group(0)
            print(match)
            first_char = match[0]
            print(first_char)
            expand_contraction = (
                CONTRACTION_MAP.get(match)
                if CONTRACTION_MAP.get(match)
                else CONTRACTION_MAP.get(match.lower())
            )
            print(expand_contraction)
            return first_char + expand_contraction[1:]
        expanded_text = CONTRACTIONS_PATTERN.sub(expand_match, tweet)
        expanded_text = re.sub("'", " ", expanded_text)
        return expanded_text
    """
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

    #tweet = "i'll eatttt y'all'd've you like  😊 a piece of cake and you'll obey me!!!! i've done that already #ihatetaylorswift"
    #print(tokenizer.tokenize_tweet(tweet))