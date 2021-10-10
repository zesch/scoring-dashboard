from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import pandas as pd

# settings for multiclass problems?

def cross_validate(df, k):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    #x = df['text'].tolist()
    #y = df['label'].astype(int).tolist()
    
    # need to convert to Unicode first - argh encodings ...
    #x = df['text'].values.astype('U')
    x = df['text'].values.astype('unicode')
    y = df['label'].astype(int).tolist()

    scores = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
    return scores

def cross_val_mat_SVM(df, k):
    results = {}
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(2,3))),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    
    # open issue: conversion to unicode...
    x = df['text'].values.astype('unicode')
    y = df['label'].astype(int).tolist()
    results['y'] = y
    results['scores'] = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
    results['y_pred'] = cross_val_predict(pipeline, x, y, cv = k, verbose=1)
    
    return results

def cross_val_dtree(df, k, list): # list as possibility to include stopwords
    results = {}
    pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer(ngram_range=(2,3))),
        ('selector',SelectKBest(k=50)),
        #('tfidf_vec', TfidfVectorizer(stop_words=list,max_features=20)),
        ('tree_clf', DecisionTreeClassifier(criterion='gini')),
        #('tree_clf', DecisionTreeClassifier(criterion='gini', max_depth=5))
    ])

    x = df['text'].values.astype('unicode')
    y = df['label'].astype(int).tolist()
    results['y'] = y
    results['scores'] = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
    results['y_pred'] = cross_val_predict(pipeline, x, y, cv = k, verbose=1)

    return results

def cross_val_reg(df, k):
    results = {}
    pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer(ngram_range=(2,3))),
        ('selector',SelectKBest(k=50)),
        ('reg_clf', LogisticRegression(solver='lbfgs')),
    ])

    x = df['text'].values.astype('unicode')
    y = df['label'].astype(int).tolist()
    results['y'] = y
    results['scores'] = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
    results['y_pred'] = cross_val_predict(pipeline, x, y, cv = k, verbose=1)

    return results

def cross_val_test(df, k):
    results_dict = {}
    results_df = pd.DataFrame()
    x = df['text'].values.astype('unicode')
    y = df['label'].astype(int).tolist()
    classifiers = {"SVC": LinearSVC(),"DecisionTree": DecisionTreeClassifier(criterion='gini'),"Regression": LogisticRegression(solver='lbfgs')}
    for key in classifiers.keys():
        pipeline = Pipeline([
            ('tfidf_vec', TfidfVectorizer(ngram_range=(2,3))),
            ('clf', classifiers[key]),
        ])
        scores = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
        results_dict[str(key)] = scores
        results_df[str(key)+ " score"] = results_dict[str(key)]
    return results_df

# select k best features als hyperparameter für DT, Reg??

 
