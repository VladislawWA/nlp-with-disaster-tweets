import fastapi
from fastapi import FastAPI
import numpy as np
import pandas as pd
import matplotlib as plt
import os
from typing import Annotated
from pydantic import BaseModel
import uvicorn

import nltk
from nltk.stem import WordNetLemmatizer

import sklearn.feature_extraction as fe
from sklearn.naive_bayes import MultinomialNB

base_dir = 'D:/Users/Vladislav/Documents/Jupyter Notebook/Лабораторные работы/Программная инженерия'
train_data = pd.read_csv(base_dir + '/nlp-getting-started/train.csv')
test_data = pd.read_csv(base_dir + '/nlp-getting-started/test.csv')
stop_words = nltk.corpus.stopwords.words('english')

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


@app.post('/predict')
async def pred(
        tweet: str,
):
    tweet_pred = tweet
    tweet = tweet_processing(tweet_pred)
    pred = clf.predict(tweet)

    return {"predict": f"{pred[0]}"}