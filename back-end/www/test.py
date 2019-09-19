import sys
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from svm_learner import SvmLearner
from lstm_learner import LSTMLearner
from late_fusion import LateFusion
from fusion_learner import FuseLearner

# Test model performance
def main(argv):
    if len(argv) < 3:
        print("Usage: python test.py [method] [model_path]")
        return
    method = argv[1]
    model_path = argv[2]
    if method is None or model_path is None:
        print("Usage: python test.py [method] [model_path]")
        return
    test(method=method, model_path=model_path)

def test(method=None, model_path=None):
    if method == "i3d-rgb":
        model = I3dLearner(mode="rgb")
        model.predict(p_model=model_path)
    elif method == "i3d-flow":
        model = I3dLearner(mode="flow")
        model.predict(p_model=model_path)
    elif method == "ts-rgb":
        model = TsLearner(mode="rgb")
        model.predict(p_model=model_path)
    elif method == "ts-flow":
        model = TsLearner(mode="flow")
        model.predict(p_model=model_path)
    elif method == "avg":
        model = LateFusion()
        model.predict(rgb_model=model_path, flow_model="../data/saved_ts/flow/first_train/2295.pt")
    elif method == "fuse":
        model = FuseLearner()
        model.predict(p_model=model_path)
    elif method == "lstm":
        model = LSTMLearner()
        model.predict(p_model=model_path)
    elif method == "svm-rgb":
        model = SvmLearner(mode="rgb")
        model.predict(p_model=model_path)
    elif method == "svm-flow":
        model = SvmLearner(mode="flow")
        model.predict(p_model=model_path)
    else:
        print("Method not allowed")
        return

if __name__ == "__main__":
    main(sys.argv)
