import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.renorm = nn.LocalResponseNorm(size=2)

        self.conv1 = nn.conv3d(in_channels=2, out_channels=96, kernel=7, stride=2)
        self.conv2 = nn.conv3d(in_channels=96, out_channels=256, kernel=5, stride=2)
        self.conv3 = nn.conv3d(in_channels=256, out_channels=512, kernel=3, stride=1)
        self.conv4 = nn.conv3d(in_channels=512, out_channels=512, kenel=3, stride=1)
        self.conv5 = nn.conv3d(in_channels=512, out_channels=512, kenel=3, stride=1)

        self.full_conn1 = nn.Linear(in_features=self.in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.renorm(x)

        x = self.conv2(x)
        self.pool(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pool(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches
        self.in_dim = x[1]

        x = self.full_conn1(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.dropout(x)

        x = self.full_conn3(x)

        return F.softmax(x, dim=0)

class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=3, stride=2)
        self.renorm = nn.LocalResponseNorm(size=2)

        self.conv1 = nn.conv3d(in_channels=3, out_channels=96, kernel=7, stride=2)
        self.conv2 = nn.conv3d(in_channels=96, out_channels=256, kernel=5, stride=2)
        self.conv3 = nn.conv3d(in_channels=256, out_channels=512, kernel=3, stride=1)
        self.conv4 = nn.conv3d(in_channels=512, out_channels=512, kenel=3, stride=1)
        self.conv5 = nn.conv3d(in_channels=512, out_channels=512, kenel=3, stride=1)

        self.full_conn1 = nn.Linear(in_features=self.in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = self.renorm(x)

        x = self.conv2(x)
        self.pool(x)
        x = F.relu(x)
        x = self.renorm(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pool(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches
        self.in_dim = x[1]

        x = self.full_conn1(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.dropout(x)

        x = self.full_conn3(x)

        return F.softmax(x, dim=0)


