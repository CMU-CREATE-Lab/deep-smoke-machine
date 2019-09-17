from base_learner import BaseLearner
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoFeatureDataset
import joblib
import uuid
from util import *
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import re


# SVM learner using I3D features
class SvmLearner(BaseLearner):
    def __init__(self,
            C=4, # SVM parameters
            mode="rgb", # can be "rgb" or "flow"
            p_feat_rgb="../data/i3d_features_rgb/", # path to load rgb feature
            p_feat_flow="../data/i3d_features_flow/", # path to load optical flow feature
            p_frame_rgb="../data/rgb/", # path to load rgb frame
            p_frame_flow="../data/flow/", # path to load optical flow frame
            p_metadata_train="../data/metadata_train.json", # path to load metadata for training
            p_metadata_validation="../data/metadata_validation.json", # path to load metadata for validation
            p_metadata_test="../data/metadata_test.json" # path to load metadata for testing
            ):
        super().__init__()

        self.C = C
        self.mode = mode
        self.p_feat_rgb = p_feat_rgb
        self.p_feat_flow = p_feat_flow
        self.p_frame_rgb = p_frame_rgb
        self.p_frame_flow = p_frame_flow
        self.p_metadata_train = p_metadata_train
        self.p_metadata_validation = p_metadata_validation
        self.p_metadata_test = p_metadata_test

    def log_parameters(self):
        text = ""
        text += "C: " + str(self.C) + "\n"
        text += "mode: " + str(self.mode) + "\n"
        text += "p_feat_rgb: " + self.p_feat_rgb + "\n"
        text += "p_feat_flow: " + self.p_feat_flow + "\n"
        text += "p_frame_rgb: " + self.p_frame_rgb + "\n"
        text += "p_frame_flow: " + self.p_frame_flow + "\n"
        text += "p_metadata_train: " + self.p_metadata_train + "\n"
        text += "p_metadata_validation: " + self.p_metadata_validation + "\n"
        text += "p_metadata_test: " + self.p_metadata_test
        self.log(text)

    def set_dataloader(self, metadata_path, root_dir):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoFeatureDataset(metadata_path=metadata_path[phase], root_dir=root_dir)
            dataloader[phase] = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=False)
        return dataloader

    def fit(self,
            save_model_path="../data/saved_svm/[model_id]/model/", # path to save the models ([model_id] will be replaced)
            save_tensorboard_path="../data/saved_svm/[model_id]/run/", # path to save data ([model_id] will be replaced)
            save_log_path="../data/saved_svm/[model_id]/log/train.log" # path to save log files ([model_id] will be replaced)
            ):
        # Set path
        model_id = str(uuid.uuid4())[0:7] + "-svm-" + self.mode
        save_model_path = save_model_path.replace("[model_id]", model_id)
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        p_feat = self.p_feat_rgb if self.mode == "rgb" else self.p_feat_flow
        p_frame = self.p_frame_rgb if self.mode == "rgb" else self.p_frame_flow

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("Start training model: " + model_id)
        self.log("save_model_path: " + save_model_path)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)
        self.log_parameters()

        # Set model
        model = SVC(C=self.C, gamma="scale")
        #model = LinearSVC(C=self.C, max_iter=10)

        # Load datasets
        metadata_path = {"train": self.p_metadata_train, "validation": self.p_metadata_validation}
        dataloader = self.set_dataloader(metadata_path, p_feat)

        # Create tensorboard writter
        writer_t = SummaryWriter(save_tensorboard_path + "/train/")
        writer_v = SummaryWriter(save_tensorboard_path + "/validation/")

        # Train and validate
        for phase in ["train", "validation"]:
            self.log("phase " + phase)
            for d in dataloader[phase]:
                file_name = d["file_name"]
                feature = d["feature"].numpy()
                true_labels = d["label"].numpy()
                if phase == "train":
                    model.fit(feature, true_labels)
                pred_labels = model.predict(feature)
            # Save precision, recall, and f-score to the log
            self.log(classification_report(true_labels, pred_labels))
            # Add video summary to tensorboard
            cm = confusion_matrix_of_samples(true_labels, pred_labels)
            writer = writer_t if phase == "train" else writer_v
            write_video_summary(writer, cm, file_name, p_frame, global_step=1)

        # Save model
        self.save(model, save_model_path + "model.pkl")

        # This is a hack to give the summary writer some time to write the data
        # Without this hack, the last added video will be missing
        time.sleep(10)

        self.log("Done training")

    def predict(self,
            p_model=None, # the path to load thepreviously self-trained model
            save_tensorboard_path="../data/saved_svm/[model_id]/run/", # path to save data ([model_id] will be replaced)
            save_log_path="../data/saved_svm/[model_id]/log/test.log" # path to save log files ([model_id] will be replaced)
            ):
        # Check
        if p_model is None:
            self.log("Need to provide model path")
            return

        # Set path
        model_id = re.search(r'\b/[0-9a-fA-F]{7}-svm-(rgb|flow)/\b', p_model).group()[1:-1]
        if model_id is None:
            model_id = "unknown-model-id"
        save_tensorboard_path = save_tensorboard_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        p_feat = self.p_feat_rgb if self.mode == "rgb" else self.p_feat_flow
        p_frame = self.p_frame_rgb if self.mode == "rgb" else self.p_frame_flow

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("Start testing with mode: " + self.mode)
        self.log("save_tensorboard_path: " + save_tensorboard_path)
        self.log("save_log_path: " + save_log_path)

        # Set model
        model = self.load(p_model)

        # Load datasets
        metadata_path = {"test": self.p_metadata_test}
        dataloader = self.set_dataloader(metadata_path, p_feat)

        # Create tensorboard writter
        writer = SummaryWriter(save_tensorboard_path + "/test/")

        # Test
        for d in dataloader["test"]:
            file_name = d["file_name"]
            feature = d["feature"].numpy()
            true_labels = d["label"].numpy()
            pred_labels = model.predict(feature)

        # Save precision, recall, and f-score to the log
        self.log(classification_report(true_labels, pred_labels))

        # Add video summary to tensorboard
        cm = confusion_matrix_of_samples(true_labels, pred_labels)
        write_video_summary(writer, cm, file_name, p_frame)

        # This is a hack to give the summary writer some time to write the data
        # Without this hack, the last added video will be missing
        time.sleep(10)

        self.log("Done testing")

    def save(self, model, out_path):
        if model is not None and out_path is not None:
            self.log("Save model to " + out_path)
            check_and_create_dir(out_path)
            joblib.dump(model, out_path)

    def load(self, in_path):
        if in_path is not None:
            self.log("Load model from " + in_path)
            model = joblib.load(in_path)
            return model
        else:
            return None
