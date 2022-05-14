import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


data1 = pd.read_csv("./data/kaggle_twitter_data.csv")
data2 = pd.read_csv("./data/dataworld_twitter_data.csv")

# Remove other columns
data1 = data1[["sentiment", "tweet"]]
data2 = data2[["sentiment", "tweet"]]

# Remove sentiment=2 from Kaggle data set
data1 = data1[data1.sentiment != 2]

frames = [data1, data2]

# All data
data = pd.concat(frames, ignore_index=True)

# Pre-processing
# Drop rows with NA or NAN
data = data.dropna()

df = data["tweet"]
# Make tweet to str
df = df.apply(str)

# Removing RT and link
df = df.apply(lambda x: re.sub(r"\bRT\b", "", x).strip())
df = df.apply(lambda x: re.sub(r"\blink\b", "", x).strip())

# Remove contractions
def contractions(s):
    s = re.sub(r"won't", " will not", s)
    s = re.sub(r"would't", " would not", s)
    s = re.sub(r"could't", " could not", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"can\'t", " can not", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve/co", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s


df = df.apply(lambda x: contractions(x))

# Remove URLS
df = df.apply(
    lambda x: re.sub(r"http?://[A-Za-z0-9./]+", "", x, flags=re.MULTILINE)
)
df = df.apply(
    lambda x: re.sub(r"https?://[A-Za-z0-9./]+", "", x, flags=re.MULTILINE)
)
df = df.apply(
    lambda x: re.sub(r"www?://[A-Za-z0-9./]+", "", x, flags=re.MULTILINE)
)

# Remove @ and #
df = df.apply(lambda x: re.sub(r"@[A-Za-z0-9_]+", "", x))
df = df.apply(lambda x: re.sub(r"#[A-Za-z0-9_]+", "", x))

# Lowercase all words
df = df.apply(lambda x: x.lower())

# Remove non-English characters
df = df.apply(lambda x: x.encode("ascii", "ignore").decode())

# Remove stopwords
stop = stopwords.words("english")
df = df.apply(lambda x: " ".join([x for x in x.split() if x not in stop]))

# Remove numbers
df = df.apply(lambda x: re.sub(r"[0-9]+", "", x))

# Remove punctuations
df = df.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

# Remove punctuations
df = df.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

# Tokenized
tokenizer = TweetTokenizer()
tokenizedTweets = []

for x in range(len(df)):
    if x is not None:
        tokenizedTweets.append(tokenizer.tokenize(df[x]))

# Lemmatize
lemmatizer = WordNetLemmatizer()

for tweet in tokenizedTweets:
    print(tweet)

    for word in tweet:
        word = lemmatizer.lemmatize(word)
    print(tweet)
    break

# ADD ANY PRE-PROCESSING PROCESSES

processed = tokenizedTweets

# Append changed tweet to database
final = []
for x in range(len(processed)):
    final.append(" ".join(processed[x]))

out = pd.DataFrame(final)

data["changedtweet"] = out
print(data.head())

# 5000 samples per label
positive = data[data["sentiment"] == 1][:5000]
negative = data[data["sentiment"] == -1][:5000]
neutral = data[data["sentiment"] == 0][:5000]

features = ["changedtweet"]
targets = ["sentiment"]
X_train = pd.DataFrame(columns = features)
X_test = pd.DataFrame(columns = features)
y_train = pd.DataFrame(columns = targets)
y_test = pd.DataFrame(columns = targets)

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for category in (positive, negative, neutral):
    X = category["changedtweet"]
    y = category["sentiment"]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y, random_state=0, train_size=0.5)
    
    X_train_list.append(Xs_train)
    X_test_list.append(Xs_test)
    y_train_list.append(ys_train)
    y_test_list.append(ys_test)


X_train = pd.concat(X_train_list, ignore_index=True)
X_test = pd.concat(X_test_list, ignore_index=True)
y_train = pd.concat(y_train_list, ignore_index=True)
y_test = pd.concat(y_test_list, ignore_index=True)
    
print(X_train.head())
print(y_train.head())