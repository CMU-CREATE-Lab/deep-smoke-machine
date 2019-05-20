from base_learner import BaseLearner
from model.pytorch_i3d import InceptionI3d

# Two-Stream Inflated 3D ConvNet learner
# https://arxiv.org/abs/1705.07750
class I3dLearner(BaseLearner):
    def __init__(self):
        super().__init__()
        self.create_logger(log_path="I3dLearner.log")
        self.log("Use Two-Stream Inflated 3D ConvNet learner")

    def fit(self, Xt, Yt, Xv=None, Yv=None):
        pass

    def predict(self, X):
        pass
