import torch
import torch.nn as nn
import numpy as np

from model.timeception.nets import timeception_pytorch
from model.pytorch_i3d import InceptionI3d, Unit3D


# I3D + Timeception
class InceptionI3dTc(nn.Module):

    def __init__(self, input_size, num_classes=2, in_channels=3, final_endpoint="Mixed_5c",
            num_tc_layers=2, dropout_keep_prob=0.5):
        super(InceptionI3dTc, self).__init__()

        # I3D input has shape (10, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        dummy_input = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print(dummy_input.size())

        # I3D
        self.i3d = InceptionI3d(num_classes=num_classes, in_channels=in_channels, final_endpoint=final_endpoint)
        self.i3d.build()

        # I3D output has shape (10, 1024, 5, 7, 7)
        i3d_dummy_output = self.i3d(dummy_input)
        print(i3d_dummy_output.size())

        # Timeception
        self.tc = timeception_pytorch.Timeception(i3d_dummy_output.size(), n_layers=num_tc_layers)

        # Timeception output has shape (10, 1600, 1, 7, 7)
        tc_dummy_output = self.tc(i3d_dummy_output)
        print(tc_dummy_output.size())

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=tc_dummy_output.size(1), output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def forward(self, x):
        x = self.i3d(x)
        x = self.tc(x)
        # logits output has shape (batch, classes, time), which is what we want to work with
        # I3D logits shape is (10, 2, 4, 1, 1)
        x = self.logits(self.dropout(self.avg_pool(x)))
        print(x.size())
        return x
