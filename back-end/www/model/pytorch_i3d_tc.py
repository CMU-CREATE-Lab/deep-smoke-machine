import torch
import torch.nn as nn
import numpy as np
from model.pytorch_i3d import InceptionI3d, Unit3D
from model.timeception.nets import timeception_pytorch


# I3D + Timeception
# Timeception for Complex Action Recognition
# https://arxiv.org/abs/1812.01289
class InceptionI3dTc(nn.Module):

    def __init__(self, input_size, num_classes=2, in_channels=3, num_tc_layers=2, dropout_keep_prob=0.5, freeze_i3d=False):
        super(InceptionI3dTc, self).__init__()
        print("Initialize the I3D+Timeception model...")
        print("num_tc_layers: " + str(num_tc_layers))
        print("freeze_i3d: " + str(freeze_i3d))
        self.freeze_i3d = freeze_i3d

        # Set the first dimension of the input size to be 1, to reduce the amount of computation
        input_size[0] = 1

        # I3D input has shape (1, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        a = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print("Input size:")
        print("\t", a.size())

        # I3D
        self.i3d = InceptionI3d(num_classes=num_classes, in_channels=in_channels)
        if freeze_i3d:
            print("Freeze I3D model")
            self.i3d.train(False)

        # I3D output has shape (1, 1024, 5, 7, 7)
        b = self.i3d(a, no_logits=True)
        print("I3D model output size:")
        print("\t", b.size())

        # Timeception
        self.tc = timeception_pytorch.Timeception(b.size(), n_layers=num_tc_layers)

        # Timeception output has shape (1, 1600, 1, 7, 7)
        c = self.tc(b)
        print("Timeception model output size:")
        print("\t", c.size())

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits_in_channels = c.size(1)
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        d = self.logits(self.dropout(self.avg_pool(c))).squeeze(3).squeeze(3)
        print("Final layer output size:")
        print("\t", d.size())

    def get_i3d_model(self):
        return self.i3d

    def replace_logits(self, num_classes):
        self.i3d.replace_logits(num_classes)
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def delete_i3d_logits(self):
        print("Delete logits in the I3D model...")
        del self.i3d.logits
        del self.i3d.avg_pool
        del self.i3d.dropout

    def forward(self, x):
        x = self.i3d(x, no_logits=True)
        x = self.tc(x)
        # Logit output has shape (10, 2, 1, 1, 1)
        # Final output has shape (batch, classes, time), which is what we want to work with
        x = self.logits(self.dropout(self.avg_pool(x))).squeeze(3).squeeze(3)
        return x
