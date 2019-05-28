import torch
from base_learner import BaseLearner
from model.pytorch_ts import *
from torch.utils.data import DataLoader
from smoke_video_dataset import SmokeVideoDataset

# Two-Stream ConvNet learner
# http://papers.nips.cc/paper/5353-two-stream-convolutional
class TsLearner(BaseLearner):
    def __init__(self):
        super().__init__()
        self.create_logger(log_path="TsLearner.log")
        self.log("Use Two-Stream ConvNet learner")

    def fit(self,
            mode="rgb",
            p_metadata_train="../data/metadata_train.json",
            p_metadata_validation="../data/metadata_validation.json",
            p_vid="../data/videos/"):

        self.log("="*60)
        self.log("="*60)
        self.log("Start training...")

    def predict(self, X):
        self.log("predict")
        pass
