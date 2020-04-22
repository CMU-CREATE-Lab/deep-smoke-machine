import sys
from util import *
from i3d_learner import I3dLearner
from r2d_learner import R2dLearner
from svm_learner import SvmLearner


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
    for i in range(len(argv)-2):
        # Can chain at most three model paths
        # BUG: when putting too many model paths at once, creating dataloader becomes very slow
        test(method=method, model_path=argv[2+i])


def test(method=None, model_path=None):
    if method == "i3d-rgb":
        model = I3dLearner(mode="rgb")
        model.test(p_model=model_path)
    elif method == "i3d-flow":
        model = I3dLearner(mode="flow")
        model.test(p_model=model_path)
    elif method == "i3d-rgb-cv-1":
        cv("rgb", "i3d", model_path, augment=True, perturb=False)
    elif method == "i3d-rgb-cv-2":
        cv("rgb", "i3d", model_path, augment=False, perturb=False)
    elif method == "i3d-rgb-cv-3":
        cv("rgb", "i3d", model_path, augment=True, perturb=True)
    elif method == "i3d-flow-cv-1":
        cv("flow", "i3d", model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-rgb-cv-1":
        cv("rgb", "i3d-ft-tc", model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-tsm-rgb-cv-1":
        cv("rgb", "i3d-ft-tc-tsm", model_path, augment=True, perturb=False)
    elif method == "i3d-tc-rgb-cv-1":
        cv("rgb", "i3d-tc", model_path, augment=True, perturb=False)
    elif method == "i3d-tsm-rgb-cv-1":
        cv("rgb", "i3d-tsm", model_path, augment=True, perturb=False)
    elif method == "i3d-nl-rgb-cv-1":
        cv("rgb", "i3d-nl", model_path, augment=True, perturb=False)
    elif method == "i3d-ft-lstm-rgb-cv-1":
        cv("rgb", "i3d-ft-lstm", model_path, augment=True, perturb=False)
    elif method == "r2d-rgb-cv-1":
        cv("rgb", "r2d", model_path, augment=True, perturb=False)
    elif method == "svm-rgb":
        model = SvmLearner(mode="rgb")
        model.test(p_model=model_path)
    elif method == "svm-rgb-cv-1":
        cv("rgb", "svm", model_path)
    elif method == "svm-flow":
        model = SvmLearner(mode="flow")
        model.test(p_model=model_path)
    elif method == "svm-flow-cv-1":
        cv("flow", "svm", model_path)
    else:
        print("Method not allowed")
        return


# Cross validation of i3d and svm model
def cv(mode, method, model_path, augment=True, perturb=False):
    if perturb:
        p_frame_rgb = "../data/rgb_perturb/"
    else:
        p_frame_rgb = "../data/rgb/"
    if method == "i3d":
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb)
    elif method == "i3d-ft-tc-tsm":
        # Use i3d model weights to finetune extra layers (with TSM)
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_tsm=True, use_tc=True, freeze_i3d=True)
    elif method == "i3d-ft-tc":
        # Use i3d model weights to finetune extra layers
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_tc=True, freeze_i3d=True)
    elif method == "i3d-tc":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_tc=True, freeze_i3d=False)
    elif method == "i3d-tsm":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_tsm=True, freeze_i3d=False)
    elif method == "i3d-nl":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_nl=True, freeze_i3d=False)
    elif method == "i3d-ft-lstm":
        # Use i3d model weights to finetune extra layers
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb,
                use_lstm=True, freeze_i3d=True)
    elif method == "r2d":
        model = R2dLearner(mode=mode)
    elif method == "svm":
        model = SvmLearner(mode=mode)
    else:
        print("Method not allowed.")
        return
    model.test(p_model=model_path)


if __name__ == "__main__":
    main(sys.argv)
