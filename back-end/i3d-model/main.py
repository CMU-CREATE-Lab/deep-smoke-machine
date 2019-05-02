import sys, os
sys.path.append(os.path.abspath(os.path.join("..", "util")))

from getData import *

def main(argv):
    data = getData()
    print(data)

if __name__ == "__main__":
    main(sys.argv)
