import sys
from util import *

def main(argv):
    # Build dataset and save it to a file
    print("Build dataset...")
    build_dataset("../data/dataset.json")
    print("Done.")

if __name__ == "__main__":
    main(sys.argv)
