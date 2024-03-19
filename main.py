import fastapi
from fastapi import FastAPI
import numpy as np
import pandas as pd
from typing import Annotated
from pydantic import BaseModel
import uvicorn

import nltk
from nltk.stem import WordNetLemmatizer

import sklearn
import sklearn.feature_extraction as fe
from sklearn.naive_bayes import MultinomialNB

train_data = pd.read_csv('./nlp-getting-started/train.csv')
test_data = pd.read_csv('./nlp-getting-started/test.csv')
stop_words = ['the', 'a', 'at', 'an', 'on',  'i', 'we', 'he', 'she', 'they']

count_vec = fe.text.CountVectorizer(stop_words=stop_words)
tfidf_transformer = fe.text.TfidfTransformer()
lemmatizer = WordNetLemmatizer()

lemm_train_data = [lemmatizer.lemmatize(word) for word in train_data['text']]
train_tok = count_vec.fit_transform(lemm_train_data)
tf_train_tok = tfidf_transformer.fit_transform(train_tok)

clf = MultinomialNB().fit(tf_train_tok, train_data['target'])


def tweet_processing(tweet):
    lemma = [lemmatizer.lemmatize(word) for word in [tweet]]

    tokens = count_vec.transform(lemma)
    tfidf_tokens = tfidf_transformer.transform(tokens)

    return tfidf_tokens


def predict(tweet):
    pred = clf.predict(tweet)

    return pred


app = FastAPI()


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
