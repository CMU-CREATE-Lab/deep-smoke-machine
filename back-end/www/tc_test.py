import numpy as np
import torch
from model.timeception.nets import timeception_pytorch
from model.pytorch_i3d_tc import InceptionI3dTc


def load(model, in_path):
    if model is not None and in_path is not None:
        print("Load model weights from " + in_path)
        try:
            model.load_state_dict(torch.load(in_path))
        except:
            print("Weights were from nn.DataParallel or DistributedDataParallel...")
            print("Remove 'module.' prefix from state_dict keys...")
            state_dict = torch.load(in_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            model.load_state_dict(new_state_dict)


# (batch_size, channel, time, height, width)
input_size = [10, 3, 36, 224, 224]
x = torch.tensor(np.zeros(input_size), dtype=torch.float32)
model = InceptionI3dTc(input_size, num_classes=400, in_channels=3, num_tc_layers=2)
model_path = "../data/pretrained_models/i3d_rgb_imagenet_kinetics.pt"
load(model.get_i3d_model(), model_path)
model.replace_logits(2)
model_path = "../data/saved_i3d/paper_result/full-augm-rgb/5c9e65a-i3d-rgb-s0/model/2047.pt"
load(model.get_i3d_model(), model_path)
model.delete_i3d_logits()
print(model(x).size())
