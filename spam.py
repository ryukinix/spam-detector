#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Copyright © 2019 Manoel Vilela
#
#    @project: Spam Detector
#     @author: Manoel Vilela
#      @email: manoel_vilela@engineer.com
#

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import scikitplot as skplt
import matplotlib.pyplot as plt

df = pd.read_csv('spam-dataset.csv', sep='\t', names=['spam', 'text'])
df.spam = df.spam.apply(lambda x: 1 if x == 'spam' else 0)
clf = MultinomialNB()
vectorizer = CountVectorizer()
p = Pipeline([('text2vec', vectorizer),
              ('classifier', clf)])
X = df.text
y = df.spam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
p.fit(X_train, y_train)
y_pred = p.predict(X_test)
acc = accuracy_score(y_test, y_pred)
skplt.metrics.plot_confusion_matrix(['SPAM' if y == 1 else 'INBOX' for y in y_test],
                                    ['SPAM' if y == 1 else 'INBOX' for y in y_pred],
                                    title=f"Matriz de Confusão, acurácia={acc:.2f}")
fig = plt.gcf()
fig.savefig("conf_matrix.png")
