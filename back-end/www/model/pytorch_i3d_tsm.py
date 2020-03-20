import torch
import torch.nn as nn
import numpy as np
from model.tsm.ops.temporal_shift import TemporalShift
from model.pytorch_i3d import InceptionI3d, Unit3D, InceptionModule


# I3D + TSM
# TSM: Temporal Shift Module for Efficient Video Understanding
# https://arxiv.org/abs/1811.08383
class InceptionI3dTsm(nn.Module):

    def __init__(self, input_size, num_classes=2, in_channels=3, dropout_keep_prob=0.5, random=False):
        super(InceptionI3dTsm, self).__init__()
        print("Initialize the I3D+TSM model...")
        print("random_temporal_shift: " + str(random))
        self.random = random

        # Set the first dimension of the input size to be 1, to reduce the amount of computation
        input_size[0] = 1

        # I3D input has shape (batch_size, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        a = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print("Input size:")
        print("\t", a.size())

        # I3D
        self.i3d = InceptionI3d(num_classes=num_classes, in_channels=in_channels)

        # I3D output has shape (batch_size, 1024, 5, 7, 7)
        b = self.i3d(a, no_logits=True)
        print("I3D model output size:")
        print("\t", b.size())

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits_in_channels = b.size(1)
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        d = self.logits(self.dropout(self.avg_pool(b))).squeeze(3).squeeze(3)

        # Final output has shape (batch_size, num_classes, time)
        print("Final layer output size:")
        print("\t", d.size())

    def add_tsm_in_inception(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, InceptionModule):
                self.add_tsm_before_conv3d(child)

    def add_tsm_before_conv3d(self, model):
        for child_name, child in model.named_children():
            if isinstance(child, nn.Conv3d):
                if child.kernel_size != [1, 1, 1]:
                    if child.in_channels >= 16:
                        print("Add tsm to: %r" % child)
                        m = TemporalShift(child, n_segment=None, n_div=16, is_video=True, random=self.random)
                        setattr(model, child_name, m)
            else:
                self.add_tsm_before_conv3d(child)

    def add_tsm_to_i3d(self):
        self.add_tsm_in_inception(self.i3d)

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
        x = self.logits(self.dropout(self.avg_pool(x))).squeeze(3).squeeze(3)
        return x
