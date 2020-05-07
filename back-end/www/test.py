import sys
from util import *
from i3d_learner import I3dLearner
from cnn_learner import CnnLearner
from svm_learner import SvmLearner


# This is the main script for model testing
# For detailed usage, run terminal command "sh bg.sh"
def main(argv):
    if len(argv) < 3:
        print("Usage: python test.py [method] [model_path]")
        return
    method = argv[1]
    model_path = argv[2]
    if method is None or model_path is None:
        print("Usage: python test.py [method] [model_path]")
        return
    for i in range(len(argv)-2):
        # Can chain at most three model paths
        # BUG: when putting too many model paths at once, creating dataloader becomes very slow
        test(method=method, model_path=argv[2+i])


def test(method=None, model_path=None):
    if method == "i3d-rgb-cv-1":
        cv("rgb", "i3d", model_path, augment=True, perturb=False)
    elif method == "i3d-rgb-cv-2":
        cv("rgb", "i3d", model_path, augment=False, perturb=False)
    elif method == "i3d-rgb-cv-3":
        cv("rgb", "i3d", model_path, augment=True, perturb=True)
    elif method == "i3d-flow-cv-1":
        cv("flow", "i3d", model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-rgb-cv-1":
        cv("rgb", "i3d-ft-tc", model_path, augment=True, perturb=False)
    elif method == "i3d-tc-rgb-cv-1":
        cv("rgb", "i3d-tc", model_path, augment=True, perturb=False)
    elif method == "i3d-tsm-rgb-cv-1":
        cv("rgb", "i3d-tsm", model_path, augment=True, perturb=False)
    elif method == "i3d-nl-rgb-cv-1":
        cv("rgb", "i3d-nl", model_path, augment=True, perturb=False)
    elif method == "i3d-ft-lstm-rgb-cv-1":
        cv("rgb", "i3d-ft-lstm", model_path, augment=True, perturb=False)
    elif method == "i3d-rgbd-cv-1":
        cv("rgbd", "i3d", model_path, augment=True, perturb=False)
    elif method == "cnn-rgb-cv-1":
        cv("rgb", "cnn", model_path, augment=True, perturb=False)
    elif method == "cnn-ft-tc-rgb-cv-1":
        cv("rgb", "cnn-ft-tc", model_path, augment=True, perturb=False)
    elif method == "svm-rgb-cv-1":
        cv("rgb", "svm", model_path)
    elif method == "svm-flow-cv-1":
        cv("flow", "svm", model_path)
    else:
        print("Method not allowed")
        return


# Cross validation of different models
def cv(mode, method, model_path, augment=True, perturb=False):
    # Set the path for loading video frames
    if perturb:
        # Use frame perturbation, where video frames are randomly shuffled
        if mode == "rgb":
            p_frame = "../data/rgb_perturb/"
        elif mode == "rgbd":
            p_frame = "../data/rgbd_perturb/"
        elif mode == "flow":
            p_frame = "../data/flow_perturb/"
    else:
        # Use the original video frames
        if mode == "rgb":
            p_frame = "../data/rgb/"
        elif mode == "rgbd":
            p_frame = "../data/rgbd/"
        elif mode == "flow":
            p_frame = "../data/flow/"

    # Set the model based on the desired method
    # The training script "train.py" has descriptions about these methods
    if method == "i3d":
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame)
    elif method == "i3d-ft-tc":
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_tc=True, freeze_i3d=True)
    elif method == "i3d-tsm":
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_tsm=True, freeze_i3d=False)
    elif method == "i3d-nl":
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_nl=True, freeze_i3d=False)
    elif method == "i3d-ft-lstm":
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_lstm=True, freeze_i3d=True)
    elif method == "cnn":
        model = CnnLearner(mode=mode, augment=augment, p_frame=p_frame,
                method="cnn", freeze_cnn=False)
    elif method == "cnn-ft-tc":
        model = CnnLearner(mode=mode, augment=augment, p_frame=p_frame,
                method="cnn-tc", freeze_cnn=True)
    elif method == "svm":
        model = SvmLearner(mode=mode)
    else:
        print("Method not allowed.")
        return
    model.test(p_model=model_path)


if __name__ == "__main__":
    main(sys.argv)
