"""
This file contains all the model related functions

"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(data_path: str, model_out_path:str) -> None:
    """
    Create and trains a classification model using the iris dataset

    Args:
        data_path: The data location (in csv format)
        model_out_path: Where to save the model object

    """
    data = pd.read_csv(data_path)
    X = data.drop(columns='target')
    y = data.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=300)
    mod = LogisticRegression(max_iter=200)
    mod.fit(X_tr, y_tr)
    
    with open(model_out_path, 'wb') as f:
        pickle.dump(mod, f)
    
    y_hat = mod.predict(X_te)
    acc = accuracy_score(y_te, y_hat)
    print(acc)

def predict(model_path: str, data_path: str, pred_out_path: str) -> None:
    """
    Uses a pre-trained model to predict the iris labels

    Args:
        model_path: The location of the trained model
        data_path: The location of the data with the features
        pred_out_path: Where to save the predictions
    
    """
    X = pd.read_csv(data_path)
    with open(model_path, 'rb') as f:
        mod = pickle.load(f)

    y_hat = mod.predict(X)
    
    np.savetxt(pred_out_path, y_hat)


