from climl.model import train, predict
from climl.utils.data import gen_iris
import fire

def main():
    """
    Here we leverage on the package fire that allows us to call
    our functions directly from the command line
    
    """
    fire.Fire({
        'train': train,
        'predict': predict,
        'sample-data': gen_iris
     })

if __name__ == '__main__':
    main()