"""
This file contains all the model related functions

"""

import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from climl.utils.exceptions import PathMissing


def train(data_path: str | None = None, model_out_path: str | None = None) -> None:
    """
    Creates and trains a classification model using the iris dataset

    Args:
        data_path: The data location (in csv format), if not provided it will
          attempt to read from stdin
        model_out_path: Where to save the model object in pickle format, if not
          given, it will output it to stdout

    """
    if data_path is None:
        data_path = sys.stdin
    data = pd.read_csv(data_path)
    x_full = data.drop(columns="target")
    y_full = data.target
    x_train, _, y_train, _ = train_test_split(
        x_full, y_full, test_size=0.25, random_state=300
    )
    mod = LogisticRegression(max_iter=200)
    mod.fit(x_train, y_train)

    if model_out_path is None:
        pickle.dump(mod, sys.stdout.buffer)
    else:
        with open(model_out_path, "wb") as fconn:
            pickle.dump(mod, fconn)


def predict(
    model_path: str | None = None,
    data_path: str | None = None,
    pred_out_path: str | None = None,
) -> None:
    """
    Uses a pre-trained model to predict the iris labels

    Args:
        model_path: The location of the trained model, if not provided it will
          attempt to read from stdin
        data_path: The location of the data with the features, if not provided
          will attempt to read from stdin
        pred_out_path: Where to save the predictions, if not provided, it will
          output to the console

    """
    if data_path is None and model_path is None:
        raise PathMissing("At least one path must be given")
    if data_path is None:
        data_path = sys.stdin
    x_pred = pd.read_csv(data_path)
    with open(model_path, "rb") as fconn:
        mod = pickle.load(fconn)

    y_hat = mod.predict(x_pred)

    if pred_out_path is None:
        pred_out_path = sys.stdout

    np.savetxt(pred_out_path, y_hat)
