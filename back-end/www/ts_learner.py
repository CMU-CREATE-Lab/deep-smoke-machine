from base_learner import BaseLearner
from model.pytorch_ts import *

# Two-Stream ConvNet learner
# http://papers.nips.cc/paper/5353-two-stream-convolutional
class TsLearner(BaseLearner):
    def __init__(self):
        super().__init__()
        self.create_logger(log_path="TsLearner.log")
        self.log("Use Two-Stream ConvNet learner")

    def fit(self, Xt, Yt, Xv=None, Yv=None):
        pass

    def predict(self, X):
        pass
