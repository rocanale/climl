"""climl entry point"""

import fire

from climl.model import train, predict
from climl.utils.data import gen_iris


def main():
    """
    Here we leverage on the package fire that allows us to call
    our functions directly from the command line

    """
    fire.Fire({"train": train, "predict": predict, "datagen": gen_iris})


if __name__ == "__main__":
    main()
