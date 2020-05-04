import numpy as np
import torch
from model.tsm.ops.models import TSN
from model.pytorch_i3d_tc import InceptionI3dTc
from model.pytorch_i3d_tsm import InceptionI3dTsm
from model.pytorch_i3d_lstm import InceptionI3dLstm
from model.pytorch_i3d_nl import InceptionI3dNl
from model.pytorch_cnn import Cnn
from model.pytorch_cnn_tc import CnnTc
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


def test_model(method="tc"):
    input_size = [batch_size, 3, time, 224, 224]
    x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
    # Test for creating object
    if method == "tc":
        model = InceptionI3dTc(input_size, num_classes=400, in_channels=3)
    elif method == "tsm":
        model = InceptionI3dTsm(input_size, num_classes=400, in_channels=3)
    elif method == "lstm":
        model = InceptionI3dLstm(input_size, num_classes=400, in_channels=3)
    elif method == "nl":
        model = InceptionI3dNl(input_size, num_classes=400, in_channels=3)
    elif method == "cnn":
        model = Cnn(input_size)
    elif method == "cnn-tc":
        model = CnnTc(input_size)
    else:
        raise NotImplementedError("Method not implemented")
    # Test for loading model
    dl = DummyLearner()
    if method not in ["cnn", "cnn-tc"]:
        model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
        dl.load(model.get_i3d_model(), model_path)
        model.replace_logits(2)
        model_path = "../data/saved_i3d/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/2047.pt"
        dl.load(model.get_i3d_model(), model_path)
        model.delete_i3d_logits()
        if method == "tsm":
            model.add_tsm_to_i3d()
        elif method == "nl":
            model.add_nl_to_i3d()
    else:
        model_path = "../data/saved_cnn/paper_result/full-augm-rgb-cnn/ce58dec-cnn-rgb-s0/model/2047.pt"
        if method == "cnn":
            dl.load(model, model_path)
        elif method == "cnn-tc":
            dl.load(model, model_path)
            model.replace_logits(2)
    print(model)
    print(model(x).size())


def test_tsn():
    x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
    model = TSN(2, time, "RGB", base_model="resnet50", dropout=0.5, is_shift=True, shift_div=8, non_local=True)
    model_path = "../data/pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth"
    dl = DummyLearner()
    dl.load(model, model_path, ignore_fc=True)
    print(model(x).size())


test_model(method="tc")
#test_model(method="tsm")
#test_model(method="nl")
#test_model(method="lstm")
#test_model(method="cnn")
#test_model(method="cnn-tc")
