import pandas as pd
import ast
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

def get_top_25_genres(dataset):
    genres = dataset['track_genres']
    
    L = []
    for i in range(len(genres)):
        L += genres[i]
        
    items,counts = np.unique(L,return_counts=True)
    genres_sorted = pd.Series(counts, index = items).sort_values(ascending=False)
    top_25_genres = list(genres_sorted.iloc[:25].index)
    
    return top_25_genres


def get_y(dataset):
    genre_index = joblib.load('metric\genre_index.dict')
    mlb = MultiLabelBinarizer(classes=list(genre_index.keys()))
    matrix_y = mlb.fit_transform(dataset['track_genres'])
    y = pd.DataFrame(matrix_y, columns = mlb.classes_)
    return y


def split_dataset(path):
    dataset = pd.read_csv(path, sep=';').drop(['Unnamed: 0', 'index', 'track_uri'], axis=1)
    dataset['track_genres'] = dataset['track_genres'].apply(ast.literal_eval)
    top_25_genres = get_top_25_genres(dataset)
    
    y = get_y(dataset).loc[:, top_25_genres]
    X = dataset.drop(['track_genres'], axis=1)
    
    X_local, y_local, X_online, y_online = iterative_train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.33)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X_local, y_local, test_size=0.30)
    X_online, y_online = pd.DataFrame(X_online, columns=X.columns), pd.DataFrame(y_online, columns=y.columns)
    X_train, y_train = pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(y_train, columns=y.columns)
    X_test, y_test = pd.DataFrame(X_test, columns=X.columns), pd.DataFrame(y_test, columns=y.columns)
    
    X_train.to_csv('data/X_train.csv', sep=';')
    y_train.to_csv('data/y_train.csv', sep=';')
    X_test.to_csv('data/X_test.csv', sep=';')
    y_test.to_csv('data/y_test.csv', sep=';')
    X_online.to_csv('data/X_online.csv', sep=';')
    y_online.to_csv('data/y_online.csv', sep=';')


split_dataset('data/spotify_dataset.csv')