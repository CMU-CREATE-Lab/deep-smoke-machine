import torch
import torch.nn as nn
import torchvision
import numpy as np
from model.pytorch_i3d import Unit3D
from model.timeception.nets import timeception_pytorch


# 2D ResNet + Timeception
class R2dTc(nn.Module):

    def __init__(self, input_size, num_classes=2, num_tc_layers=1, dropout_keep_prob=0.5):
        super().__init__()
        print("Initialize 2D ResNet")

        # Set the first dimension of the input size to be 4, to reduce the amount of computation
        input_size[0] = 4

        # Input has shape (batch_size, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        a = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print("Input size:")
        print("\t", a.size())

        # 2D CNN (we use ResNet18)
        b = a.transpose(1, 2) # (batch_size, time, channel, height, width)
        bs = b.size()
        b = b.reshape(bs[0]*bs[1], bs[2], bs[3], bs[4]) # (batch_size X time, channel, height, width)
        self.cnn = torchvision.models.resnet18(pretrained=True, progress=True)
        b = self.cnn(b) # (batch_size X time, num_features)
        print("CNN model output size:")
        print("\t", b.size())

        # Timeception
        c = b.reshape(bs[0], bs[1], -1) # (batch_size, time, num_features)
        cs = c.size()
        c = c.reshape(cs[0], cs[1], cs[2], 1, 1) # (batch_size, time, num_features, 1, 1)
        c = c.transpose(1, 2) # (batch_size, num_features, time, 1, 1)
        self.tc = timeception_pytorch.Timeception(c.size(), n_layers=num_tc_layers)
        c = self.tc(c) # (batch_size, 1240, 18, 1, 1) if num_tc_layers=1
        print("Timeception model output size:")
        print("\t", c.size())

        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 1, 1], stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits_in_channels = c.size(1)
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        d = self.logits(self.dropout(self.avg_pool(c))).squeeze(3).squeeze(3) # (batch, num_classes, time)
        print("Final layer output size:")
        print("\t", d.size())

    def get_cnn_model(self):
        return self.cnn

    def replace_logits(self, num_classes):
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def delete_cnn_fc(self):
        print("Delete the final fully connected layer in CNN...")
        del self.cnn.fc

    def forward(self, x):
        # x has shape (batch_size, channel, time, height, width)
        x = x.transpose(1, 2) # (batch_size, time, channel, height, width)
        xs = x.size()
        x = x.reshape(xs[0]*xs[1], xs[2], xs[3], xs[4]) # (batch_size X time, channel, height, width)
        x = self.cnn(x) # (batch_size X time, num_features)
        x = x.reshape(xs[0], xs[1], -1) # (batch_size, time, num_features)
        xs = x.size()
        x = x.reshape(xs[0], xs[1], xs[2], 1, 1) # (batch_size, time, num_features, 1, 1)
        x = x.transpose(1, 2) # (batch_size, num_features, time, 1, 1)
        x = self.tc(x) # (batch_size, 1240, 18, 1, 1) if num_tc_layers=1
        x = self.logits(self.dropout(self.avg_pool(x))).squeeze(3).squeeze(3) # (batch, num_classes, time)
        return x
