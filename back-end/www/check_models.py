import numpy as np
import torch
from model.tsm.ops.models import TSN
from model.pytorch_i3d_tc import InceptionI3dTc
from model.pytorch_i3d_tsm import InceptionI3dTsm
from model.pytorch_i3d_lstm import InceptionI3dLstm
from collections import OrderedDict
from base_learner import BaseLearner


class DummyLearner(BaseLearner):
    def fit(self):
        pass
    def test(self):
        pass


# (batch_size, channel, time, height, width)
batch_size = 4
time = 36
input_size = [batch_size, 3, time, 224, 224]


def test_model(model="tc"):
    x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
    if model == "tc":
        model = InceptionI3dTc(input_size, num_classes=400, in_channels=3)
    elif model == "tsm":
        model = InceptionI3dTsm(input_size, num_classes=400, in_channels=3, random=True)
    elif model == "tsm-disabled":
        model = InceptionI3dTsm(input_size, num_classes=400, in_channels=3, random=True, enable_tsm=False)
    elif model == "lstm":
        model = InceptionI3dLstm(input_size, num_classes=400, in_channels=3)
    else:
        raise NotImplementedError("Model not implemented")
    dl = DummyLearner()
    model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
    dl.load(model.get_i3d_model(), model_path)
    model.replace_logits(2)
    model_path = "../data/saved_i3d/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/2047.pt"
    dl.load(model.get_i3d_model(), model_path)
    model.delete_i3d_logits()
    print(model(x).size())


def test_tsn():
    x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
    model = TSN(2, time, "RGB", base_model="resnet50", dropout=0.5, is_shift=True, shift_div=8, non_local=True)
    model_path = "../data/pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth"
    dl = DummyLearner()
    dl.load(model, model_path, ignore_fc=True)
    print(model(x).size())

print("="*60)
test_model(model="tc")
print("="*60)
test_model(model="tsm")
print("="*60)
test_model(model="tsm-disabled")
print("="*60)
test_model(model="lstm")
