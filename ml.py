from re import A

import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn import model_selection
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np

# settings for multiclass problems?

def cross_validate(df, k, col_label, col_text):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    #x = df['text'].tolist()
    #y = df['label'].astype(int).tolist()
    
    # need to convert to Unicode first - argh encodings ...
    #x = df['text'].values.astype('U')
    x = df[col_text].values.astype('unicode')
    y = df[col_label].astype(int).tolist()

    scores = cross_val_score(pipeline, x, y, cv = k, scoring='f1_micro', error_score="raise")
    return scores

    # pipeline = Pipeline([
    #     ('vect', CountVectorizer()),#ngram_range=(2,3))),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', LinearSVC())
    # ])

    # pipeline = Pipeline([
    #     ('tfidf_vec', TfidfVectorizer()), #(ngram_range=(2,3))),
    #     #('selector',SelectKBest(k=50)),
    #     #('tfidf_vec', TfidfVectorizer(stop_words=list,max_features=20)),
    #     ('tree_clf', DecisionTreeClassifier(criterion='gini')),
    #     #('tree_clf', DecisionTreeClassifier(criterion='gini', max_depth=5))
    # ])

    # pipeline = Pipeline([
    #     ('tfidf_vec', TfidfVectorizer()), #ngram_range=(2,3))),
    #     #('selector',SelectKBest(k=50)),
    #     ('reg_clf', LogisticRegression(solver='lbfgs')),
    # ])

@st.cache
def cross_val_test(df, k, col_text, col_label):
    results_dict = {}
    results_df = pd.DataFrame()
    x = df[col_text].values.astype('unicode')
    y = df[col_label].astype(int).tolist()
    classifiers = {"SVM": LinearSVC(),
                   "DecisionTree": DecisionTreeClassifier(criterion='gini'),
                   "RandomForest":RandomForestClassifier(criterion='gini'),
                   "Regression": LogisticRegression(solver='lbfgs')
                   }
    for key in classifiers.keys():
        pipeline = Pipeline([
            ('tfidf_vec', TfidfVectorizer(ngram_range=(1,3))),
            ('clf', classifiers[key]),
        ])
        scores = cross_val_score(pipeline, x, y, cv = k, scoring='f1_macro', error_score="raise")
        results_dict[str(key)] = scores
        results_df[str(key)+ " score"] = results_dict[str(key)]
    return results_df

# select k best features als hyperparameter für DT, Reg??

# ------------------------------------------------------------------------------------------------

def flatten(list):
    return [item for sublist in list for item in sublist]

# thought about adding loss, but: predict_probability is not defined for all classifiers (not for linearSVC e.g.) and 
# otherwise log loss does not work on multiple classes

def get_clf(clf):
    if clf == 'SVM':
        return LinearSVC()
    elif clf == 'Regression':
        return LogisticRegression(solver='lbfgs')
    elif clf == 'Decision Tree':
        return DecisionTreeClassifier(criterion='gini')
    elif clf == 'Random Forest':
        return RandomForestClassifier(criterion='gini')

@st.cache
def test(clf, df, k, col_text, col_label):
    df.index = df.index - 1 #needed since index +1 for visualization
    container = {}
    all_y_test = []
    all_y_pred = []

    av_accuracy = []
    av_precision = []
    av_recall = []
    av_fscore = []

    x = pd.Series(df[col_text].values.astype('unicode'))
    y = pd.Series(df[col_label].astype(int))

    kfold = model_selection.KFold(n_splits=k,random_state=7,shuffle=True)
    pipeline = Pipeline([
    ('vect', CountVectorizer()),#ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('clf', clf)
    ])
    i = 0
    for train_idx, test_idx in kfold.split(x):
        i+=1
        #st.write(train_idx)
        #st.write(test_idx)
        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for index in train_idx:
            x_train.append(x[index])
            y_train.append(y[index])

        for index in test_idx:
            x_test.append(x[index])
            y_test.append(y[index])

        model = pipeline.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        av_accuracy.append(accuracy_score(y_test,y_pred))
        av_precision.append(precision_score(y_test,y_pred,average='macro'))
        av_recall.append(recall_score(y_test,y_pred,average='macro'))
        av_fscore.append(f1_score(y_test,y_pred,average='macro'))    
        
        y_test = np.asarray(y_test)
        all_y_pred.append(y_pred)
        all_y_test.append(y_test)

    all_y_test = flatten(all_y_test)
    all_y_pred = flatten(all_y_pred)

    container['accuracy'] = av_accuracy
    container['precision'] = av_precision
    container['recall'] = av_recall
    container['f1_score'] = av_fscore
    container['y_pred'] = all_y_pred
    container['y_true'] = all_y_test

    return container