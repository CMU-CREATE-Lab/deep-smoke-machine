import sys, os
from util import *

def main(argv):
    dataset = build_dataset("../data/dataset.json")
    print(dataset)

if __name__ == "__main__":
    main(sys.argv)
