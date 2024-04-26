import sys
from i3d_learner import I3dLearner
from cnn_learner import CnnLearner
from svm_learner import SvmLearner


# This is the main script for model training and validating
# For detailed usage, run terminal command "sh bg.sh"
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
    # Description of the methods are in the cv function
    if method == "i3d-rgb-cv-1":
        # This is the "RGB-I3D" model in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-rgb-cv-2":
        # This is the "RGB-I3D-ND" model in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=False, perturb=False)
    elif method == "i3d-rgb-cv-3":
        # This is the "RGB-I3D-FP" model in our AAAI paper
        # To run this model, you need to run the following command to perturb frames first
        # python perturb_frames.py
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d", model_path=model_path, augment=True, perturb=True)
    elif method == "i3d-flow-cv-1":
        # This is the "Flow-I3D" model in our AAAI paper
        # To run this model, you need to compute optical flow frames first
        # Go to the process_videos.py file and change flow_type to 1
        # Then, run the follwing again
        # python process_videos.py
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_flow_imagenet_kinetics.pt"
        cv("flow", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-ft-tc-rgb-cv-1":
        # This is the "RGB-TC" model in our AAAI paper
        # To run this model, you need to run the i3d-rgb-cv-1 method first to get the best models
        # Then, change the following path to point to the best models
        if model_path is None:
            model_path = [
                    "../data/pretrained_models/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/682.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/549f8df-i3d-rgb-s1/model/1176.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/a8a7205-i3d-rgb-s2/model/679.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/58474a0-i3d-rgb-s4/model/591.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/5260727-i3d-rgb-s5/model/585.pt"]
        cv("rgb", "i3d-ft-tc", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-tc-rgb-cv-1":
        # This model is not used in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-tc", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-tsm-rgb-cv-1":
        # This is the "RGB-TSM" model in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-tsm", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-nl-rgb-cv-1":
        # This is the "RGB-NL" model in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgb", "i3d-nl", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-ft-lstm-rgb-cv-1":
        # This is the "RGB-LSTM" model in our AAAI paper
        # To run this model, you need to run the i3d-rgb-cv-1 method first to get the best models
        # Then, change the following path to point to the best models
        if model_path is None:
            model_path = [
                    "../data/pretrained_models/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/682.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/549f8df-i3d-rgb-s1/model/1176.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/a8a7205-i3d-rgb-s2/model/679.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/55563e4-i3d-rgb-s3/model/573.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/58474a0-i3d-rgb-s4/model/591.pt",
                    "../data/pretrained_models/paper_result/full-augm-rgb/5260727-i3d-rgb-s5/model/585.pt"]
        cv("rgb", "i3d-ft-lstm", model_path=model_path, augment=True, perturb=False)
    elif method == "i3d-rgbd-cv-1":
        # This model is not used in our AAAI paper
        if model_path is None:
            model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        cv("rgbd", "i3d", model_path=model_path, augment=True, perturb=False)
    elif method == "svm-rgb-cv-1":
        # This is the "RGB-SVM" model in our AAAI paper
        # To run this model, you need to run the following command to extract i3d-rgb features first
        # python extract_features.py i3d-rgb
        cv("rgb", "svm")
    elif method == "svm-flow-cv-1":
        # This is the "Flow-SVM" model in our AAAI paper
        # To run this model, you need to compute optical flow frames (see the i3d-flow-cv-1 method above)
        # Then, run the following command to extract i3d-flow features
        # python extract_features.py i3d-flow
        cv("flow", "svm")
    else:
        print("Method not allowed")
        return


# Cross validation of different models
# mode="rgb" means using the rgb channels
# mode="flow" means using the optical flow channels
# mode="rgbd" means using the rgb and dark channel (see "compute_dark_channel.py")
def cv(mode, method, model_path=None, augment=True, perturb=False):
    # Set the path for loading video frames and features
    if mode == "rgb":
        p_feat = "../data/i3d_features_rgb/"
    elif mode == "flow":
        p_feat = "../data/i3d_features_flow/"
    elif mode == "rgbd":
        p_feat = "../data/i3d_features_rgbd/"
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
    if method == "i3d":
        # Use Kinetics pretrained weights to train the baseline I3D model with Inception-v1 layers
        # https://arxiv.org/abs/1705.07750
        num_steps_per_update = 2
        init_lr = 0.1
        milestones = [500, 1500]
        if mode == "rgbd":
            num_steps_per_update = 1
            init_lr = 0.2
            milestones = [1000, 2000]
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                init_lr=init_lr, num_steps_per_update=num_steps_per_update, milestones=milestones)
    elif method == "i3d-tc":
        # Use Kinetics pretrained weights to train the entire network with Timeception layers
        # https://arxiv.org/abs/1812.01289
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_tc=True, freeze_i3d=False, batch_size_train=8,
                milestones=[1000, 2000], num_steps_per_update=1)
    elif method == "i3d-ft-tc":
        # Use I3D model self-trained weights to finetune extra Timeception layers
        # https://arxiv.org/abs/1812.01289
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_tc=True, freeze_i3d=True, batch_size_train=8,
                milestones=[1000, 2000], num_steps_per_update=1)
    elif method == "i3d-tsm":
        # Use Kinetics pretrained weights to train the entire network with Temporal Shift Module
        # https://arxiv.org/abs/1811.08383
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_tsm=True, freeze_i3d=False,
                milestones=[1000, 2000], weight_decay=0.0000000001, num_steps_per_update=1)
    elif method == "i3d-nl":
        # Use Kinetics pretrained weights to train the entire network with Non-Local Blocks
        # https://arxiv.org/abs/1711.07971
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_nl=True, freeze_i3d=False)
    elif method == "i3d-ft-lstm":
        # Use I3D model self-trained weights to finetune LSTM layers
        # https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.8.1735
        model = I3dLearner(mode=mode, augment=augment, p_frame=p_frame,
                use_lstm=True, freeze_i3d=True, batch_size_train=8,
                milestones=[1000, 2000], num_steps_per_update=1, weight_decay=0.0001)
    elif method == "cnn":
        # Use ImageNet pretrained weights to train the 2D CNN model
        # https://arxiv.org/abs/1502.03167
        model = CnnLearner(mode=mode, augment=augment, p_frame=p_frame,
                method="cnn", freeze_cnn=False)
    elif method == "cnn-ft-tc":
        # Use 2D CNN model self-trained weights to finetune extra Timeception layers
        model = CnnLearner(mode=mode, augment=augment, p_frame=p_frame,
                method="cnn-tc", freeze_cnn=True,
                milestones=[1000, 2000], num_steps_per_update=1)
    elif method == "svm":
        # Support vector machine
        model = SvmLearner(mode=mode, p_feat=p_feat)
    else:
        print("Method not allowed.")
        return

    # Set the pretrained model paths for all dataset splits
    if type(model_path) is not list:
        model_path = [model_path]*6

    # Cross validation on the 5th split by camera
    model.fit(p_model=model_path[5],
            model_id_suffix="-s5",
            p_metadata_train="../data/split/metadata_train_split_4_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_4_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_4_by_camera.json")
    # Cross validation on the 4th split by camera
    model.fit(p_model=model_path[4],
            model_id_suffix="-s4",
            p_metadata_train="../data/split/metadata_train_split_3_by_camera.json",
            p_metadata_validation="../data/split/metadata_validation_split_3_by_camera.json",
            p_metadata_test="../data/split/metadata_test_split_3_by_camera.json")
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


if __name__ == "__main__":
    main(sys.argv)
