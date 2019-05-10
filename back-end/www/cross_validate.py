import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from util import *

# Cross validation of extracted features
def main(argv):
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
    model = SVC()
    model.fit(X["train"], y["train"])

    # Evaluation
    y_predict = {}
    for phase in ["validation", "test"]:
        y_predict[phase] = model.predict(X[phase])
        print("Phase", phase)
        print(classification_report(y[phase], y_predict[phase]))

if __name__ == "__main__":
    main(sys.argv)
