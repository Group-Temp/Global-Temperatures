import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from string import punctuation

data1 = pd.read_csv('../data/kaggle_twitter_data.csv')
data2 = pd.read_csv('../data/dataworld_twitter_data.csv')

#Remove other columns
data1 = data1[['sentiment','tweet']]
data2 = data2[['sentiment','tweet']]

#Remove sentiment=2 from Kaggle data set
data1 = data1[data1.sentiment != 2]

frames = [data1,data2]

#All data
data = pd.concat(frames, ignore_index=True)

#Pre-processing
#Drop rows with NA or NAN
data = data.dropna()

#Make tweet to str
data['tweet']=data['tweet'].apply(str)

#Removing RT and link
data['tweet']=data['tweet'].apply(lambda x: re.sub(r'\bRT\b','',x).strip())
data['tweet']=data['tweet'].apply(lambda x: re.sub(r'\blink\b','',x).strip())

#Remove contractions
def contractions(s):
    s = re.sub(r"won't", " will not",s)
    s = re.sub(r"would't", " would not",s)
    s = re.sub(r"could't", " could not",s)
    s = re.sub(r"\'d", " would",s)
    s = re.sub(r"can\'t", " can not",s)
    s = re.sub(r"n\'t", " not", s)
    s= re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    return s
data['tweet']=data['tweet'].apply(lambda x:contractions(x))

#Remove URLS
data['tweet']=data['tweet'].apply(lambda x: re.sub(r"((www.[^s]+)|(https?://[^s]+)|(http?://[^s]+))", '', x, flags=re.MULTILINE))

#Remove @ and #
data['tweet']=data['tweet'].apply(lambda x: re.sub(r'@[A-Za-z0-9_]+','', x))
data['tweet']=data['tweet'].apply(lambda x: re.sub(r'#[A-Za-z0-9_]+','', x))

#Lowercase all words
data['tweet']=data['tweet'].apply(lambda x: x.lower())

#Remove non-English characters
data['tweet']=data['tweet'].apply(lambda x:x.encode("ascii", "ignore").decode())

#Remove stopwords
stop = stopwords.words('english')
data['tweet']=data['tweet'].apply(lambda x: " ".join([x for x in x.split() if x not in stop]))

#Remove numbers
data['tweet']=data['tweet'].apply(lambda x: re.sub(r'[0-9]+', '', x))

#Remove punctuations
data['tweet']=data['tweet'].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))

#Tokenized
tokenizer=TweetTokenizer()
tokenizedWords=[]

for x in range(len(data['tweet'])):
    if x is not None:
        tokenizedWords.append(tokenizer.tokenize(data['tweet'][x]))

print(tokenizedWords[1])

positive = data[data['sentiment']==1]

negative = data[data['sentiment']==-1]

neutral = data[data['sentiment']==0]


