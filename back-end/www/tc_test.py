import numpy as np
import torch as T
from model.timeception.nets import timeception_pytorch
from model.pytorch_i3d_tc import InceptionI3dTc

# (batch_size, channel, time, height, width)
input_size = (10, 3, 36, 224, 224)
x = T.tensor(np.zeros(input_size), dtype=T.float32)
model = InceptionI3dTc(input_size, num_classes=2, in_channels=3, final_endpoint="Mixed_5c", num_tc_layers=2)
model(x)
