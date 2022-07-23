"""This module generates / retrieves data"""

import sys

from sklearn.datasets import load_iris


def gen_iris(output_path: str | None = None, only_inference: bool = False) -> None:
    """
    Generates the iris data set containing four feature columns
    plus the target

    Args:
        output_path: Where to save the file, if omited it will print
        it to standard output
        only_inference: Whether to skip the target column

    """
    if output_path is None:
        output_path = sys.stdout
    iris_data, iris_labels = load_iris(as_frame=True, return_X_y=True)
    if not only_inference:
        iris_data["target"] = iris_labels
    iris_data.to_csv(output_path, index=False)
