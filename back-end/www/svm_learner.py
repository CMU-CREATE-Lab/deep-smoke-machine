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
import torch
import time
import re
import shutil


# SVM learner using I3D features
class SvmLearner(BaseLearner):
    def __init__(self,
            C=1, # SVM parameters
            mode="rgb", # can be "rgb" or "flow"
            p_feat="../data/i3d_features_rgb/", # path to load features
            ):
        super().__init__()

        self.C = C
        self.mode = mode
        self.p_feat = p_feat

    def log_parameters(self):
        text = ""
        text += "C: " + str(self.C) + "\n"
        text += "mode: " + str(self.mode) + "\n"
        text += "p_feat: " + self.p_feat + "\n"
        self.log(text)

    def set_dataloader(self, metadata_path, root_dir):
        dataloader = {}
        for phase in metadata_path:
            self.log("Create dataloader for " + phase)
            dataset = SmokeVideoFeatureDataset(metadata_path=metadata_path[phase], root_dir=root_dir)
            dataloader[phase] = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=False)
        return dataloader

    def fit(self,
            p_model=None, # not used, just for consistency with the i3d model's parameters
            model_id_suffix="", # the suffix appended after the model id
            p_metadata_train="../data/split/metadata_train_split_0_by_camera.json", # metadata path (train)
            p_metadata_validation="../data/split/metadata_validation_split_0_by_camera.json", # metadata path (validation)
            p_metadata_test="../data/split/metadata_test_split_0_by_camera.json", # metadata path (test)
            save_model_path="../data/saved_svm/[model_id]/model/", # path to save the models ([model_id] will be replaced)
            save_log_path="../data/saved_svm/[model_id]/log/train.log", # path to save log files ([model_id] will be replaced)
            save_metadata_path="../data/saved_svm/[model_id]/metadata/" # path to save metadata ([model_id] will be replaced)
            ):
        # Set path
        model_id = str(uuid.uuid4())[0:7] + "-svm-" + self.mode
        model_id += model_id_suffix
        save_model_path = save_model_path.replace("[model_id]", model_id)
        save_log_path = save_log_path.replace("[model_id]", model_id)
        save_metadata_path = save_metadata_path.replace("[model_id]", model_id)

        # Copy training, validation, and testing metadata
        check_and_create_dir(save_metadata_path)
        shutil.copy(p_metadata_train, save_metadata_path + "metadata_train.json")
        shutil.copy(p_metadata_validation, save_metadata_path + "metadata_validation.json")
        shutil.copy(p_metadata_test, save_metadata_path + "metadata_test.json")

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("save_model_path: " + save_model_path)
        self.log("save_log_path: " + save_log_path)
        self.log("p_metadata_train: " + p_metadata_train)
        self.log("p_metadata_validation: " + p_metadata_validation)
        self.log("p_metadata_test: " + p_metadata_test)
        self.log_parameters()

        # Set model
        model = SVC(C=self.C, gamma="scale")
        #model = LinearSVC(C=self.C, max_iter=10)

        # Load datasets
        metadata_path = {"train": p_metadata_train, "validation": p_metadata_validation}
        dataloader = self.set_dataloader(metadata_path, self.p_feat)

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

        # Save model
        self.save(model, save_model_path + "model.pkl")

        self.log("Done training")

    def test(self,
            p_model=None # the path to load thepreviously self-trained model
            ):
        # Check
        if p_model is None:
            self.log("Need to provide model path")
            return

        # Set path
        match = re.search(r'\b/[0-9a-fA-F]{7}-svm-(rgb|flow)[^/]*/\b', p_model)
        model_id = match.group()[1:-1]
        if model_id is None:
            self.log("Cannot find a valid model id from the model path.")
            return
        p_root = p_model[:match.start()] + "/" + model_id + "/"
        p_metadata_test = p_root + "metadata/metadata_test.json" # metadata path (test)
        save_log_path = p_root + "log/test.log" # path to save log files

        # Set logger
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("Start testing with mode: " + self.mode)
        self.log("save_log_path: " + save_log_path)
        self.log("p_metadata_test: " + p_metadata_test)
        self.log_parameters()

        # Set model
        model = self.load(p_model)

        # Load datasets
        metadata_path = {"test": p_metadata_test}
        dataloader = self.set_dataloader(metadata_path, self.p_feat)

        # Test
        for d in dataloader["test"]:
            file_name = d["file_name"]
            feature = d["feature"].numpy()
            true_labels = d["label"].numpy()
            pred_labels = model.predict(feature)

        # Save precision, recall, and f-score to the log
        self.log(classification_report(true_labels, pred_labels))

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
