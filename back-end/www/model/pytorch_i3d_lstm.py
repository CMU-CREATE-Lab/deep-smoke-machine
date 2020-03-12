import torch
import torch.nn as nn
import numpy as np
from model.pytorch_i3d import InceptionI3d, Unit3D
from base_learner import Reshape


# I3D + LSTM
class InceptionI3dLstm(nn.Module):

    def __init__(self, input_size, num_classes=2, in_channels=3, dropout_keep_prob=0.5, freeze_i3d=False):
        super(InceptionI3dLstm, self).__init__()
        print("Initialize the I3D+LSTM model...")

        # Set the first dimension of the input size to be 1, to reduce the amount of computation
        input_size[0] = 1

        # I3D input has shape (batch_size, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        a = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print("Input size:")
        print("\t", a.size())

        # I3D
        self.i3d = InceptionI3d(num_classes=num_classes, in_channels=in_channels)
        if freeze_i3d:
            print("Freeze I3D model")
            self.i3d.train(False)

        # I3D output has shape (batch_size, 1024, 5, 7, 7)
        b = self.i3d(a, no_logits=True)
        print("I3D model output size:")
        print("\t", b.size())

        # LSTM
        bs = b.size()
        self.lstm = nn.LSTM(bs[1]*bs[3]*bs[4], 512, num_layers=1, batch_first=True)

        # LSTM output has shape (batch_size, 512, 5, 1, 1)
        self.lstm_reshape_before = Reshape((bs[2], -1))
        c = self.lstm_reshape_before(b)
        c, _ = self.lstm(c)
        cs = c.size()
        self.lstm_reshape_after = Reshape((cs[2], cs[1], 1, 1))
        c = self.lstm_reshape_after(c)
        print("LSTM model output size:")
        print("\t", c.size())

        # Logits
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits_in_channels = c.size(1)
        self.logits = Unit3D(in_channels=self.logits_in_channels, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        d = self.logits(self.dropout(c)).squeeze(3).squeeze(3)

        # Final output has shape (batch_size, num_classes, time)
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
        x = self.lstm_reshape_before(x)
        x, _ = self.lstm(x)
        x = self.lstm_reshape_after(x)
        x = self.logits(self.dropout(x)).squeeze(3).squeeze(3)
        return x
