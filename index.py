import streamlit as sl
import pandas as pd
import sklearn
import joblib

model = joblib.load("./trained_classifier.joblib")
label = "Write text here"
text = sl.text_area(label, value='', height=None, max_chars=340, key=None)

if text:
    for each in model.predict_proba([text]):
        sl.text(each)

