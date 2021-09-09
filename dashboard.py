import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import base64
from ml import cross_validate

st.set_page_config("Scoring Dashboard", None, "wide", "auto")

body, stats = st.columns([4, 1])

body.write(
"""
# Free-text Scoring Dashboard

Analyze how well your free-text data can be automatically scored
"""
)

# TODO files could have header or not
# TODO we should not rely on specific column names, but auto-detect what is probably what
# TODO take care of encodings!
uploaded_file = body.file_uploader("Upload dataset (csv format)")
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file, delimiter="\t")

if df is not None:
    st.header('Dataset Statistics')
    st.write('number of instances', str(len(df)))

    st.sidebar.header('Configuration')

    # Language
    option = st.sidebar.selectbox(
        'Which language?',
        ['German','English'])

    # Numerical Data?
    labels = st.sidebar.radio(
        "Label type?",
        ('Categorical', 'Numeric - discrete', 'Numeric continuous'))

    #Algorithm?
    algorithm = st.sidebar.radio(
        "Algorithm?",
        ('SVM', 'Regression', 'Decision Tree'))

    # ML
    st.write(df)
    scores = cross_validate(df, 5)
    st.write(scores)