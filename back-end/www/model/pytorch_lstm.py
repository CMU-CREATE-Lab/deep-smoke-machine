import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
torch.set_printoptions(threshold=sys.maxsize)

class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
        )
        mock_input = torch.randn(8, 3, 224, 224)
        mock_output = self.convolution(mock_input)
        flattened_output = torch.flatten(mock_output, start_dim=1)
        self.num_nodes = flattened_output.shape[1] # Get number of nodes from flattened value's size, then convert 0 dim tensor to integer
        fc_in_dim = self.num_nodes
        self.lstm = nn.LSTM(self.num_nodes, self.num_nodes, num_layers=2)
        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)
        #self.full_conn3 = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        # Convolution over every frame in the video for feature extraction

        b, c, f, h, w = x.shape
        features = []

        for i in range(f):
            image = x[:, :, i, :, :]
            conv = self.convolution(image)
            conv = torch.flatten(conv, start_dim=1) # Flattens layers without losing batches
            features.append(conv)

        features = torch.cat(features).view(len(features), b, -1)

        # LSTM for whole-video learning
        #self.lstm.flatten_parameters()
        output, _  = self.lstm(features)#, (self.h0, self.c0))

        # Fully connected network for classification
        output = output.permute(1,0,2)
        fc = self.full_conn1(output)
        fc = F.dropout(fc)

        fc = self.full_conn2(fc)
        fc = F.dropout(fc)

        fc = self.full_conn3(fc)

        return F.softmax(fc, dim=0)

"""
class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=108, out_channels=96, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
            #          stride=1),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.ReLU(),
        )

        mock_input = torch.randn(32, 108, 224, 224)
        mock_output = self.model(mock_input)
        flattened_output = torch.flatten(mock_output, start_dim=1)
        fc_in_dim = flattened_output.shape[1] # Get number of nodes from flattened value's size, then convert 0 dim tensor to integer


        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        #self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        #self.full_conn3 = nn.Linear(in_features=2048, out_features=2)
        self.full_conn3 = nn.Linear(in_features=4096, out_features=2)


    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches

        x = self.full_conn1(x)
        #x = F.dropout(x)

        #x = self.full_conn2(x)
        #x = F.dropout(x)

        x = self.full_conn3(x)

        return F.softmax(x, dim=0)
"""
