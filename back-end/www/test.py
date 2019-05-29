import sys
import numpy as np
from sklearn.metrics import classification_report
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from svm_learner import SvmLearner
from util import *

# Test model performance
def main(argv):
    if len(argv) < 2:
        print("Usage: python test.py [method] [model_path]")
        return
    method = argv[1]
    model_path = argv[2]
    if model is None or model_path is None:
        print("Usage: python test.py [method] [model_path]")
        return
    test(method=method, model_path=model_path)

def test(method=None, model_path=None):
    if method == "i3d":
        model = I3dLearner()
    elif method == "ts":
        model = TsLearner()
    elif method == "svm":
        model = SvmLearner()
    else:
        print("Method not allowed")
        return

    # Load model weights
    model.load(model, model_path)

    # Evaluate the model on test set

if __name__ == "__main__":
    main(sys.argv)
