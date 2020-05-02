import torch
import torch.nn as nn
import torchvision
import numpy as np


# 2D ResNet
class R2d(nn.Module):

    def __init__(self, input_size, num_classes=2):
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
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, num_classes)
        b = self.cnn(b) # (batch_size X time, num_classes)
        b = b.reshape(bs[0], bs[1], -1) # (batch_size, time, num_classes)
        b = b.transpose(1, 2) # (batch_size, num_classes, time)
        print("CNN model output size:")
        print("\t", b.size())

    def forward(self, x):
        # x has shape (batch_size, channel, time, height, width)
        x = x.transpose(1, 2) # (batch_size, time, channel, height, width)
        xs = x.size()
        x = x.reshape(xs[0]*xs[1], xs[2], xs[3], xs[4]) # (batch_size X time, channel, height, width)
        x = self.cnn(x) # (batch_size X time, num_classes)
        x = x.reshape(xs[0], xs[1], -1) # (batch_size, time, num_classes)
        x = x.transpose(1, 2) # (batch_size, num_classes, time)
        return x
