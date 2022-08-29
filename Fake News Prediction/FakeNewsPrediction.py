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
print(stopwords.words("english"))

# loading data set to pandas frame
fakenews_data=pd.read_csv("F:/ML projects/fakenews.csv")

# check number of missing values
print(fakenews_data.isnull().sum())

# replacing null values with empty string
fakenews_data=fakenews_data.fillna("")
