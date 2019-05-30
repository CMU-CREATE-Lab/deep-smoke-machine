from base_learner import BaseLearner
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import uuid
from util import *
import numpy as np

# SVM learner using I3D features
class SvmLearner(BaseLearner):
    def __init__(self,
            C=4, # SVM parameters
            save_model_path="saved_svm/"):
        super().__init__()
        self.create_logger(log_path="SvmLearner.log")
        self.log("Use SVM learner with I3D features")

        self.C = C
        self.save_model_path = save_model_path

    def fit(self,
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_feat="../data/features/"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

        # Load data
        d = {}
        d["train"] = self.dataset(p_feat, p_metadata_train)
        d["validation"] = self.dataset(p_feat, p_metadata_validation)

        # Train
        model = SVC(C=self.C, gamma="scale")
        model.fit(d["train"]["feature"], d["train"]["label"])

        # Validate
        self.log("Evaluate...")
        label_pred = {}
        for phase in ["train", "validation"]:
            label_pred[phase] = model.predict(d[phase]["feature"])
            self.log("phase: " + phase)
            self.log(classification_report(d[phase]["label"], label_pred[phase]))

        # Save
        model_id = str(uuid.uuid4())[0:7] + "-svm"
        check_and_create_dir(self.save_model_path)
        self.save(model, self.save_model_path + model_id + ".pkl")

        print("Done fit")

    def predict(self,
            p_metadata_test="../data/metadata_test.json",
            p_feat="../data/features/",
            p_model=None):

        if p_model is None:
            self.log("Need to provide model path")
            return

        self.log("="*60)
        self.log("="*60)
        self.log("Start testing...")

        model = self.load(p_model)
        d = self.dataset(p_feat, p_metadata_test)
        label_pred = model.predict(d["feature"])
        self.log(classification_report(d["label"], label_pred))

        print("Done predict")

    def save(self, model, out_path):
        if model is not None and out_path is not None:
            joblib.dump(model, out_path)

    def load(self, in_path):
        if in_path is not None:
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
