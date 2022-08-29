import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download("stopwords")

# loading data set to pandas frame
fakenews_data=pd.read_csv("F:/ML projects/fakenews.csv")

# check number of missing values
print(fakenews_data.isnull().sum())

# replacing null values with empty string
fakenews_data=fakenews_data.fillna("")
fakenews_data["content"]=fakenews_data["author"]+" "+fakenews_data["title"]

port_stem=PorterStemmer()

def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z]"," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

fakenews_data["content"]=fakenews_data["content"].apply(stemming)

X=fakenews_data["content"].values
Y=fakenews_data["label"].values

vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()

model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
training_score=accuracy_score(X_train_prediction,Y_train)

X_test_prediction=model.predict(X_test)
test_score=accuracy_score(X_test_prediction,Y_test)

print(training_score)
print(test_score)