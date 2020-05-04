import numpy as np
import torch
from model.tsm.ops.models import TSN
from model.pytorch_i3d import InceptionI3d
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
in_channels = 3

def test_model(method="i3d"):
    input_size = [batch_size, in_channels, time, 224, 224]
    x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
    # Test for creating object
    if method == "i3d":
        model = InceptionI3d(num_classes=400, in_channels=in_channels)
    elif method == "i3d-tc":
        model = InceptionI3dTc(input_size, num_classes=400, in_channels=in_channels)
    elif method == "i3d-tsm":
        model = InceptionI3dTsm(input_size, num_classes=400, in_channels=in_channels)
    elif method == "i3d-lstm":
        model = InceptionI3dLstm(input_size, num_classes=400, in_channels=in_channels)
    elif method == "i3d-nl":
        model = InceptionI3dNl(input_size, num_classes=400, in_channels=in_channels)
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
        if method == "i3d":
            dl.load(model, model_path, fill_dim=True)
        else:
            dl.load(model.get_i3d_model(), model_path)
        model.replace_logits(2)
        model_path = "../data/saved_i3d/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/2047.pt"
        if method == "i3d":
            dl.load(model, model_path, fill_dim=True)
        else:
            dl.load(model.get_i3d_model(), model_path)
            model.delete_i3d_logits()
        if method == "i3d-tsm":
            model.add_tsm_to_i3d()
        elif method == "i3d-nl":
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


test_model(method="i3d")
#test_model(method="i3d-tc")
#test_model(method="i3d-tsm")
#test_model(method="i3d-nl")
#test_model(method="i3d-lstm")
#test_model(method="cnn")
#test_model(method="cnn-tc")
