from sklearn.datasets import load_iris
import sys

def gen_iris(output_path: str = None, only_inference:bool = False) -> None:
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
    x = load_iris(as_frame=True)
    if not only_inference:
        x.data['target'] = x.target
    x.data.to_csv(output_path, index = False)
