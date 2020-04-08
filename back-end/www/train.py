import sys
from i3d_learner import I3dLearner
from svm_learner import SvmLearner


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
    elif method == "i3d-flow":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        model = I3dLearner(mode="flow")
        model.fit(p_model=model_path)
    elif method == "i3d-flow-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        cv("flow", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-rgb-cv-1":
        if model_path is None:
            model_path = [
                    "../data/saved_i3d/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/2047.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb/549f8df-i3d-rgb-s1/model/2058.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb/a8a7205-i3d-rgb-s2/model/2037.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/2005.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb/58474a0-i3d-rgb-s4/model/2068.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb/5260727-i3d-rgb-s5/model/2047.pt"]
        cv("rgb", "i3d-ft-tc", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-tsm-rgb-cv-1":
        if model_path is None:
            model_path = [
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/852e8a8-i3d-rgb-s0/model/1267.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/b93e0af-i3d-rgb-s1/model/1274.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/eb083d3-i3d-rgb-s2/model/1261.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/8cb712b-i3d-rgb-s3/model/1241.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/616fb4d-i3d-rgb-s4/model/1280.pt",
                    "../data/saved_i3d/paper_result/full-augm-rgb-tsm/b831f55-i3d-rgb-s5/model/1267.pt"]
        cv("rgb", "i3d-ft-tc-tsm", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-tc-rgb-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-tc", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-tsm-rgb-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-tsm", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-nl-rgb-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-nl", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-lstm-rgb-cv-1":
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-lstm", model_path=model_path, augment=True, perturb=False)
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
    elif method == "i3d-ft-tc-tsm":
        # Use i3d model weights to finetune extra layers
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_tsm=True, use_tc=True, freeze_i3d=True, batch_size_train=8,
                milestones_rgb=[300, 900], num_steps_per_update=1, weight_decay=0.0001)
    elif method == "i3d-ft-tc":
        # Use i3d model weights to finetune extra layers
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_tc=True, freeze_i3d=True, batch_size_train=8,
                milestones_rgb=[1000, 2000], num_steps_per_update=1)
    elif method == "i3d-tc":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_tc=True, freeze_i3d=False, batch_size_train=8)
    elif method == "i3d-tsm":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_tsm=True, freeze_i3d=False,
                milestones_rgb=[1000], weight_decay=0.00000001)
    elif method == "i3d-nl":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_nl=True, freeze_i3d=False,
                milestones_rgb=[300, 900], weight_decay=0.0001)
    elif method == "i3d-lstm":
        # Use Kinetics pretrained weights to train the entire network
        model = I3dLearner(mode=mode, augment=augment, p_frame_rgb=p_frame_rgb, p_frame_flow=p_frame_flow,
                use_lstm=True, freeze_i3d=False, batch_size_train=8)
    elif method == "svm":
        model = SvmLearner(mode=mode)
    else:
        print("Method not allowed.")
        return

    if type(model_path) is not list:
        model_path = [model_path]*6

    # Cross validation on the 1st split by camera
    model.fit(p_model=model_path[0],
            model_id_suffix="-s0",
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json")
    # Cross validation on the 2nd split by camera
    model.fit(p_model=model_path[1],
            model_id_suffix="-s1",
            p_metadata_train="../data/split/metadata_train_split_1_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_1_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_1_by_camera.json")
    # Cross validation on the 3rd split by camera
    model.fit(p_model=model_path[2],
            model_id_suffix="-s2",
            p_metadata_train="../data/split/metadata_train_split_2_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_2_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_2_by_camera.json")
    # Cross validation on the split by date
    model.fit(p_model=model_path[3],
            model_id_suffix="-s3",
            p_metadata_train="../data/split/metadata_train_split_by_date.json",
            p_metadata_validation="../data/split/metadata_validation_split_by_date.json",
            p_metadata_test="../data/split/metadata_test_split_by_date.json")
    # Cross validation on the 4th split by camera
    model.fit(p_model=model_path[4],
            model_id_suffix="-s4",
            p_metadata_train="../data/split/metadata_train_split_3_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_3_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_3_by_camera.json")
    # Cross validation on the 5th split by camera
    model.fit(p_model=model_path[5],
            model_id_suffix="-s5",
            p_metadata_train="../data/split/metadata_train_split_4_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_4_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_4_by_camera.json")


if __name__ == "__main__":
    main(sys.argv)
