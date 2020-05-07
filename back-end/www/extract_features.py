import sys
from i3d_learner import I3dLearner


# The script for extracting features
def main(argv):
    if len(argv) < 2:
        print("Usage: python extract_features.py [method]")
        print("Optional usage: python extract_features.py [method] [model_path]")
        return
    method = argv[1]
    if method is None:
        print("Usage: python extract_features.py [method]")
        print("Optional usage: python extract_features.py [method] [model_path]")
        return
    model_path = None
    if len(argv) > 2:
        model_path = argv[2]
    extract_features(method=method, model_path=model_path)


def extract_features(method=None, model_path=None):
    if method == "i3d-rgb":
        model = I3dLearner(mode="rgb")
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        model.extract_features(p_model=model_path, p_feat="../data/i3d_features_rgb/")
    elif method == "i3d-flow":
        model = I3dLearner(mode="flow")
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        model.extract_features(p_model=model_path, p_feat="../data/i3d_features_flow/")
    else:
        print("Method not allowed")
        return


if __name__ == "__main__":
    main(sys.argv)
