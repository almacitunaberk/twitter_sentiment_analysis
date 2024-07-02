import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Hyperparameters:

train_val_ratio = 0.9
max_features = 7000
max_iter = 100
vectorizer_C = 1e5


tweets = []
labels = []

data_file_path = "/Users/tunaberkalmaci/Downloads/twitter_sentiment_analysis/src/data/twitter-datasets/twitter-datasets"

def load_tweets(filename, label):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tweets.append(line.rstrip())
            labels.append(label)

pos_file = f"{data_file_path}/train_pos_full.txt"
neg_file = f"{data_file_path}/train_neg_full.txt"
load_tweets(neg_file, 0)
load_tweets(pos_file, 1)

tweets = np.array(tweets)
labels = np.array(labels)

np.random.seed(1)

shuffled_indices = np.random.permutation(len(tweets))
split_idx = int(train_val_ratio * len(tweets))
train_indices = shuffled_indices[:split_idx]
val_indices = shuffled_indices[split_idx:]

vectorizer = CountVectorizer(max_features=max_features)

X_train = vectorizer.fit_transform(tweets[train_indices])
X_val = vectorizer.transform(tweets[val_indices])

Y_train = labels[train_indices]
Y_val = labels[val_indices]

model = LogisticRegression(C=vectorizer_C, max_iter=max_iter)
model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)

train_accur = (Y_train_pred == Y_train).mean()
val_accur = (Y_val_pred == Y_val).mean()

print(f"Training accuracy: {train_accur:.05f}")
print(f"Validation accuracy: {val_accur:.05f}")

