import numpy as np
import torch
from model.tsm.ops.models import TSN
from collections import OrderedDict
from base_learner import BaseLearner


class DummyLearner(BaseLearner):
    def fit(self):
        pass
    def test(self):
        pass


# (batch_size, channel, time, height, width)
time = 36
batch_size = 8
input_size = [batch_size, 3, time, 224, 224]
x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
num_class = 2
num_segments = time # this must be equal to the time dimension
modality = "RGB"
#base_model = "resnet50"
base_model = "mobilenetv2"
model = TSN(num_class, num_segments, modality,
        base_model=base_model, dropout=0.5, is_shift=True,
        shift_div=8, shift_place="blockres", non_local=True,
        consensus_type="avg", img_feature_dim=224)
#model_path = "../data/pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth"
model_path = "../data/pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth"
dl = DummyLearner()
dl.load(model, model_path, ignore_fc=True)
print(model(x).size())
