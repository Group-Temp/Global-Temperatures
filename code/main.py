import pandas as pd
import re
import string
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from nltk.util import ngrams
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report, accuracy_score

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

# Lowercase all words
df = df.apply(lambda x: x.lower())

# Remove non-English characters
df = df.apply(lambda x: x.encode("ascii", "ignore").decode())

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

# Removing RT and link
df = df.apply(lambda x: re.sub(r"\bRT\b", "", x).strip())
df = df.apply(lambda x: re.sub(r"\blink\b", "", x).strip())

# Remove @ and #
df = df.apply(lambda x: re.sub(r"@[A-Za-z0-9_]+", "", x))
df = df.apply(lambda x: re.sub(r"#[A-Za-z0-9_]+", "", x))

# Remove contractions
df = df.apply(lambda x: contractions.fix(x))

# Remove stopwords
stop = stopwords.words("english")
df = df.apply(lambda x: " ".join([x for x in x.split() if x not in stop]))

# Remove numbers
df = df.apply(lambda x: re.sub(r"[0-9]+", "", x))

# Remove punctuations
df = df.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Tokenized
tokenizedTweets = [word_tokenize(x) for x in df]

# Lemmatize
lemmatizer = WordNetLemmatizer()
for tweet in tokenizedTweets:
    for word in tweet:
        word = lemmatizer.lemmatize(word)

# Stemming
porter_stemmer = PorterStemmer()
for tweet in tokenizedTweets:
    for word in tweet:
        word = porter_stemmer.stem(word)

# ADD ANY PRE-PROCESSING PROCESSES

processed = tokenizedTweets

# Append changed tweet to database
final = []
for x in range(len(processed)):
    final.append(" ".join(processed[x]))

out = pd.DataFrame(final)

data["changedtweet"] = out

# 5000 samples per label
positive = data[data["sentiment"] == 1][:5000]
negative = data[data["sentiment"] == -1][:5000]
neutral = data[data["sentiment"] == 0][:9000]

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

for category in (positive, negative):
    X = category["changedtweet"]
    y = category["sentiment"]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y, random_state=0, train_size=0.5)
    
    X_train_list.append(Xs_train)
    X_test_list.append(Xs_test)
    y_train_list.append(ys_train)
    y_test_list.append(ys_test)

X = neutral["changedtweet"]
y = neutral["sentiment"]
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X, y, random_state=0, train_size=0.5)
    
X_train_list.append(Xs_train)
X_test_list.append(Xs_test)
y_train_list.append(ys_train)
y_test_list.append(ys_test)


X_train = pd.concat(X_train_list, ignore_index=True)
X_test = pd.concat(X_test_list, ignore_index=True)
y_train = pd.concat(y_train_list, ignore_index=True)
y_test = pd.concat(y_test_list, ignore_index=True)

#print(len(X_train))
#print(len(X_test))
def generateWordList(wordDf, sentimentScore, n_gram_size):
    resultDf = wordDf[(wordDf['sentiment'] == sentimentScore)]

    sentences = [sentence.split() for sentence in resultDf['changedtweet']]
    wordArray = []
    for i in range(0, len(sentences)):
        wordArray += sentences[i]

    counterList = Counter(ngrams(wordArray, n_gram_size)).most_common(80)

    counterDf = pd.DataFrame()

    delimiter = ' '
    print(f"\n***N-Gram (Sentiment: {sentimentScore})")
    for i in range(0, len(counterList)):
        counterDict = {
            "N-Gram" : delimiter.join(counterList[i][0])
            ,"Occurrences" : counterList[i][1]
        }
        # convert dict to series before concating to the dataframe.
        counterDict = pd.DataFrame(counterDict, index=[0])
        counterDf = pd.concat([counterDf, counterDict], ignore_index=True)

    return counterDf

SIZE=3
counterDfNews = generateWordList(data[['sentiment','changedtweet']], -1, SIZE)
counterDfNewsTop = counterDfNews.head(20)
#print(counterDfNewsTop)

#To check for unigrams, bigrams, and trigrams, manipulate ngram_range    
Tfidf_vect = TfidfVectorizer(ngram_range=(2,2))
Tfidf_vect.fit(data['changedtweet'])
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear')
SVM.fit(Train_X_Tfidf,y_train)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print(classification_report(y_test, predictions_SVM))

# Classifier - Algorithm - Decision Tree
dt = DecisionTreeClassifier()
dt.fit(Train_X_Tfidf, y_train)
predictions_DecisionTree = dt.predict(Test_X_Tfidf)
#print("Decision Tree Accuracy Score -> ",accuracy_score(predictions_DecisionTree, y_test)*100)