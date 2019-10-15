import sys
from util import *
from i3d_learner import I3dLearner
from ts_learner import TsLearner
from svm_learner import SvmLearner
from lstm_learner import LSTMLearner
try:
    from pt_ts_learner import PtTsLearner
except ImportError:
    PtTsLearner = None
try:
    from fusion_learner import FuseLearner
except ImportError:
    FuseLearner = None
try:
    from late_fusion import LateFusion
except ImportError:
    LateFusion = None


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
        model.test(p_model=model_path)
    elif method == "i3d-flow":
        model = I3dLearner(mode="flow")
        model.test(p_model=model_path)
    elif method == "i3d-rgb-cv-1":
        # full data augmentation, from pretrained model
        i3d_cv("rgb", model_path, True)
    elif method == "i3d-flow-cv-1":
        # full data augmentation, from pretrained model
        i3d_cv("flow", model_path, True)
    elif method == "i3d-rgb-cv-2":
        # no data augmentation, from pretrained model
        i3d_cv("rgb", model_path, False)
    elif method == "i3d-flow-cv-2":
        # no data augmentation, from pretrained model
        i3d_cv("flow", model_path, False)
    elif method == "ts-rgb":
        model = TsLearner(mode="rgb")
        model.test(p_model=model_path)
    elif method == "ts-flow":
        model = TsLearner(mode="flow")
        model.test(p_model=model_path)
    elif method == "avg":
        model = LateFusion()
        model.test(rgb_model=model_path, flow_model="../data/saved_ts/flow/first_train/2295.pt")
    elif method == "fuse":
        model = FuseLearner()
        model.test(p_model=model_path)
    elif method == "lstm":
        model = LSTMLearner()
        model.test(p_model=model_path)
    elif method == "svm-rgb":
        model = SvmLearner(mode="rgb")
        model.test(p_model=model_path)
    elif method == "svm-flow":
        model = SvmLearner(mode="flow")
        model.test(p_model=model_path)
    else:
        print("Method not allowed")
        return


# Cross validation of i3d model
def i3d_cv(mode, model_path, augment):
    model = I3dLearner(mode=mode, augment=augment)
    model.test(p_model=model_path)


if __name__ == "__main__":
    main(sys.argv)
