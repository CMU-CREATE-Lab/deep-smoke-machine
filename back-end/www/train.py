import sys
import numpy as np
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from svm_learner import SvmLearner
from util import *

# Train the model
def main(argv):
    if len(argv) < 2:
        print("Usage: python train.py [method]")
        return
    method = argv[1]
    if method is None:
        print("Usage: python train.py [method]")
        return
    train(method=method)

def train(method=None):
    if method == "i3d":
        model = I3dLearner()
    elif method == "ts":
        model = TsLearner()
    elif method == "svm":
        model = SvmLearner()
    else:
        print("Method not allowed")
        return
    model.fit()

if __name__ == "__main__":
    main(sys.argv)
