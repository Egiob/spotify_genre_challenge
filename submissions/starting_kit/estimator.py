
import pandas as pd
import ast
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


def get_estimator():


    cols = [
    'danceability',
    'energy',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo', 
    'mode', 
    'duration_ms', 
    'time_signature', 
    'key',
    'loudness'
    ]

    transformer = make_column_transformer(
        ('passthrough', cols)
    )

    pipeline = make_pipeline(
        transformer,
        DecisionTreeClassifier()
    )


    return pipeline
