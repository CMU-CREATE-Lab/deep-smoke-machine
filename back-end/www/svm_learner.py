from base_learner import BaseLearner
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
import uuid
from util import *
import numpy as np

# SVM learner using I3D features
class SvmLearner(BaseLearner):
    def __init__(self,
            C=4, # SVM parameters
            mode="rgb", # can be "rgb" or "flow"
            save_model_path="../data/saved_svm/",
            p_feat_rgb="../data/i3d_features_rgb/",
            p_feat_flow="../data/i3d_features_flow/",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_metadata_test="../data/metadata_test.json"
            ):
        super().__init__()

        self.C = C
        self.mode = mode
        self.save_model_path = save_model_path
        self.p_feat_rgb = p_feat_rgb
        self.p_feat_flow = p_feat_flow
        self.p_metadata_train = p_metadata_train
        self.p_metadata_validation = p_metadata_validation
        self.p_metadata_test = p_metadata_test

    def fit(self, save_log_path="../data/saved_svm/train.log"):
        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("Start training with mode: " + self.mode)

        # Set path
        p_feat = self.p_feat_rgb if self.mode == "rgb" else self.p_feat_flow

        # Load data
        d = {}
        d["train"] = self.dataset(p_feat, self.p_metadata_train)
        d["validation"] = self.dataset(p_feat, self.p_metadata_validation)

        # Train
        model = SVC(C=self.C, gamma="scale")
        #model = LinearSVC(C=self.C, max_iter=10)
        model.fit(d["train"]["feature"], d["train"]["label"])

        # Validate
        self.log("Evaluate...")
        label_pred = {}
        for phase in ["train", "validation"]:
            label_pred[phase] = model.predict(d[phase]["feature"])
            self.log("phase: " + phase)
            self.log(classification_report(d[phase]["label"], label_pred[phase]))

        # Save
        model_id = str(uuid.uuid4())[0:7] + "-svm-" + self.mode
        check_and_create_dir(self.save_model_path)
        self.save(model, self.save_model_path + model_id + ".pkl")

        print("Done training")

    def predict(self, p_model=None, save_log_path="../data/saved_svm/test.log"):
        if p_model is None:
            self.log("Need to provide model path")
            return

        self.create_logger(log_path=save_log_path)
        self.log("="*60)
        self.log("="*60)
        self.log("Use SVM learner with I3D features")
        self.log("Start testing with mode: " + self.mode)

        p_feat = self.p_feat_rgb if self.mode == "rgb" else self.p_feat_flow
        model = self.load(p_model)
        d = self.dataset(p_feat, self.p_metadata_test)
        label_pred = model.predict(d["feature"])
        self.log(classification_report(d["label"], label_pred))

        self.log("Done testing")

    def save(self, model, out_path):
        if model is not None and out_path is not None:
            self.log("Save model to " + out_path)
            joblib.dump(model, out_path)

    def load(self, in_path):
        if in_path is not None:
            self.log("Load model from " + in_path)
            model = joblib.load(in_path)
            return model
        else:
            return None

    # Build the dataset from the metadata file
    def dataset(self, p_feat, p_metadata):
        pos = [47, 23, 19, 15]
        neg = [32, 20, 16, 12]
        metadata = load_json(p_metadata)
        feature = []
        label = []
        for v in metadata:
            feature.append(np.load(p_feat + v["file_name"] + ".npy"))
            s = v["label_state_admin"] # TODO: need to change this to label_state in the future
            if s in pos:
                label.append(1)
            elif s in neg:
                label.append(0)
            else:
                label.append(None)
        return {"feature": feature, "label": label}
