"""
This file contains all the model related functions

"""

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train(data_path: str, model_out_path: str) -> None:
    """
    Create and trains a classification model using the iris dataset

    Args:
        data_path: The data location (in csv format)
        model_out_path: Where to save the model object

    """
    data = pd.read_csv(data_path)
    x_full = data.drop(columns="target")
    y_full = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.25, random_state=300
    )
    mod = LogisticRegression(max_iter=200)
    mod.fit(x_train, y_train)

    with open(model_out_path, "wb") as fconn:
        pickle.dump(mod, fconn)

    y_hat = mod.predict(x_test)
    acc = accuracy_score(y_test, y_hat)
    print(acc)


def predict(model_path: str, data_path: str, pred_out_path: str = None) -> None:
    """
    Uses a pre-trained model to predict the iris labels

    Args:
        model_path: The location of the trained model
        data_path: The location of the data with the features
        pred_out_path: Where to save the predictions, if not provided, it will
          output to the console

    """
    x_pred = pd.read_csv(data_path)
    with open(model_path, "rb") as fconn:
        mod = pickle.load(fconn)

    y_hat = mod.predict(x_pred)

    if pred_out_path is None:
        pred_out_path = sys.stdout

    np.savetxt(pred_out_path, y_hat)
