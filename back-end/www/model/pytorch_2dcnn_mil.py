import torch
import torch.nn as nn
import torchvision
import numpy as np


# 2D CNN + Multiple Instance Learning
# Real-world Anomaly Detection in Surveillance Videos
# https://arxiv.org/pdf/1801.04264.pdf
class MIL(nn.Module):

    def __init__(self, input_size, num_classes=2):
        super(MIL, self).__init__()
        print("Initialize 2D CNN + Multiple Instance Learning...")

        # Input has shape (batch_size, 3, 36, 224, 224)
        # (batch_size, channel, time, height, width)
        a = torch.tensor(np.zeros(input_size), dtype=torch.float32)
        print("Input size:")
        print("\t", a.size())

        # Change it to have shape (batch_size X time, channel, height, width)
        b = a.transpose(1, 2) # swap time and channel
        bs = b.size() # (batch_size, time, channel, height, width)
        b = b.reshape(bs[0]*bs[1], bs[2], bs[3], bs[4])
        self.shape_before_cnn = b.size()

        # 2D CNN (here we use ResNet18)
        self.cnn = torchvision.models.resnet18(pretrained=True, progress=True)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, num_classes)

        # 2D CNN output has shape (batch_size X time, 2)
        b = self.cnn(b)
        print("CNN model output size:")
        print("\t", b.size())

        # Final output has shape (batch_size, num_classes, time)
        c = b.reshape(bs[0], bs[1], -1)
        self.shape_after_cnn = c.size()
        c = c.transpose(1, 2) # swap channel and num_classes
        print("Final layer output size:")
        print("\t", c.size())

    def forward(self, x):
        # x has shape (batch_size, channel, time, height, width)
        x = x.transpose(1, 2) # (batch_size, time, channel, height, width)
        x = x.reshape(self.shape_before_cnn) # (batch_size X time, channel, height, width)
        x = self.cnn(x) # (batch_size X time, num_classes)
        x = x.reshape(self.shape_after_cnn) # (batch_size, time, num_classes)
        x = x.transpose(1, 2) # (batch_size, num_classes, time)
        return x
