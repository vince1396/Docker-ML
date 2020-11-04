import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from stop_words import get_stop_words

df = pd.read_csv("labels.csv")
df.dropna()
df.drop_duplicates()
df['tweet'] = df['tweet'].astype('str')
clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)
clf = clf.fit(X=df['tweet'], y=df['class'])
dump(clf, 'trained_classifier.joblib')
