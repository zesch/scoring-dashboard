from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def cross_validate(df, k):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    #x = df['text'].tolist()
    #y = df['label'].astype(int).tolist()
   
    # need to convert to Unicode first - argh encodings ...
    x = df['text'].values.astype('U')
    y = df['label'].astype(int).tolist()
   
    scores = cross_val_score(pipeline, x, y, scoring='f1_micro', error_score="raise")
    return scores
