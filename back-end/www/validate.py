import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from util import *

# Fit the model and test its performance
def main(argv):
    if len(argv) < 2:
        print("Usage: python test.py [method]")
        return
    method = argv[1]
    if method == "feature":
        test_feature()
    else:
        test(method=method)

def test(method=None):
    if method == "i3d":
        model = I3dLearner()
    elif method == "ts":
        model = TsLearner()
    else:
        print("Method not allowed")
        return

    model.fit()

def test_feature():
    p = "../data/"
    p_feat = p + "features/"

    # Load data
    X = {}
    y = {}
    pos = [47, 23, 19, 15]
    neg = [32, 20, 16, 12]
    for phase in ["train", "validation", "test"]:
        metadata = load_json(p + "metadata_" + phase + ".json")
        feature = []
        label = []
        for v in metadata:
            feature.append(np.load(p_feat + v["file_name"] + ".npy"))
            s = v["label_state_admin"] # TODO: need to change this to label_state in the future
            if s in pos:
                label.append(1)
            elif s in neg:
                label.append(0)
            else:
                label.append(None)
        X[phase] = feature
        y[phase] = label

    # Training
    print("Train...")
    model = SVC(C=4, gamma="scale")
    model.fit(X["train"], y["train"])

    # Evaluation
    print("Evaluate...")
    y_predict = {}
    for phase in ["validation", "test"]:
        y_predict[phase] = model.predict(X[phase])
        print("Phase", phase)
        print(classification_report(y[phase], y_predict[phase]))
    print("Done")

if __name__ == "__main__":
    main(sys.argv)
