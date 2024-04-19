import fastapi
from fastapi import FastAPI
import pandas as pd
from typing import Annotated
from pydantic import BaseModel
from joblib import load

from nltk.stem import WordNetLemmatizer

import sklearn.feature_extraction as fe
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = FastAPI()

# load data
train_data = pd.read_csv('./nlp-getting-started/train.csv')
test_data = pd.read_csv('./nlp-getting-started/test.csv')

nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('wordnet2022')
nlp = load('en_core_web_sm')


# used lemms
lemmatizer = WordNetLemmatizer()
lemm_train_data = [lemmatizer.lemmatize(word) for word in train_data['text']]


# used pipeline
pipe = load('./pipe_data_transf.joblib')
tf_train_tok = pipe.transform(train_data)


# model
clf = MultinomialNB().fit(tf_train_tok, train_data['target'])


def tweet_processing(tweet):
    lemma = [lemmatizer.lemmatize(word) for word in [tweet]]

    tfidf_tokens = pipe.transform(tweet)

    return tfidf_tokens


def predict(tweet):
    pred = clf.predict(tweet)

    return pred


class Item(BaseModel):
    text: str


@app.post('/predict/')
def pred(tweet: Item):
    tweet_pred = tweet.text
    tweet = tweet_processing(tweet_pred)
    pred = clf.predict(tweet)[0]

    return {"predict": f"{pred}"}


@app.get('/info')
def info():
    return {"info": "Naive Bayas model"}
