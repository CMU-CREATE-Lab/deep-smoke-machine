import sys
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


# Train the model
def main(argv):
    if len(argv) < 2:
        print("Usage: python train.py [method]")
        print("Optional usage: python train.py [method] [model_path]")
        return
    method = argv[1]
    if method is None:
        print("Usage: python train.py [method]")
        print("Optional usage: python train.py [method] [model_path]")
        return
    model_path = None
    if len(argv) > 2:
        model_path = argv[2]
    train(method=method, model_path=model_path)


def train(method=None, model_path=None):
    if method == "i3d-rgb":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        model = I3dLearner(mode="rgb")
        model.fit(p_model=model_path)
    elif method == "i3d-rgb-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-rgb-cv-2":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=False, perturb=False)
    elif method == "i3d-rgb-cv-3":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=True, perturb=True)
    elif method == "i3d-rgb-cv-4":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=False, perturb=True)
    elif method == "i3d-rgb-cv-5":
        cv("rgb", "i3d", model_path=None, augment=True, perturb=False)
    elif method == "i3d-flow":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        model = I3dLearner(mode="flow")
        model.fit(p_model=model_path)
    elif method == "i3d-flow-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        cv("flow", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "ts-rgb":
        model = TsLearner(mode="rgb")
        model.fit()
    elif method == "ts-flow":
        model = TsLearner(mode="flow")
        model.fit()
    elif method == "pt-flow":
        model = PtTsLearner()
        model.fit(mode="pt-flow")
    elif method == "svm-rgb":
        model = SvmLearner(mode="rgb")
        model.fit()
    elif method == "svm-rgb-cv-1":
        cv("rgb", "svm")
    elif method == "svm-flow":
        model = SvmLearner(mode="flow")
        model.fit()
    elif method == "svm-flow-cv-1":
        cv("flow", "svm")
    elif method == "lstm":
        model = LSTMLearner()
        model.fit()
    elif method == "fuse":
        model = FuseLearner()
        model.fit()
    else:
        print("Method not allowed")
        return


# Cross validation of i3d or svm model
def cv(mode, method, model_path=None, augment=True, perturb=False):
    if perturb:
        p_frame_rgb = "../data/rgb_perturb/"
        p_frame_flow = "../data/flow_perturb/"
    else:
        p_frame_rgb = "../data/rgb/"
        p_frame_flow = "../data/flow/"
    if method == "i3d":
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow)
    elif method == "svm":
        model = SvmLearner(mode=mode)
    else:
        print("Method not allowed.")
        return
    # Cross validation on the 1st split by camera)
    model.fit(p_model=model_path,
            model_id_suffix="-s0",
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json")
    # Cross validation on the 2nd split by camera)
    model.fit(p_model=model_path,
            model_id_suffix="-s1",
            p_metadata_train="../data/split/metadata_train_split_1_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_1_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_1_by_camera.json")
    # Cross validation on the 3rd split by camera)
    model.fit(p_model=model_path,
            model_id_suffix="-s2",
            p_metadata_train="../data/split/metadata_train_split_2_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_2_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_2_by_camera.json")
    # Cross validation on the split by date
    model.fit(p_model=model_path,
            model_id_suffix="-s3",
            p_metadata_train="../data/split/metadata_train_split_by_date.json",
            p_metadata_validation="../data/split/metadata_validation_split_by_date.json",
            p_metadata_test="../data/split/metadata_test_split_by_date.json")


if __name__ == "__main__":
    main(sys.argv)
