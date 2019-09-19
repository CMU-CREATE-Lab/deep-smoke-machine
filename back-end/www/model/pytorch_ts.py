import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=72, out_channels=96, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.ReLU(),

            #nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            #nn.MaxPool2d(kernel_size=3, stride=1),
            #nn.ReLU(),
        )

        mock_input = torch.randn(32, 72, 224, 224)
        mock_output = self.model(mock_input)
        flattened_output = torch.flatten(mock_output, start_dim=1)
        fc_in_dim = flattened_output.shape[1] # Get number of nodes from flattened value's size, then convert 0 dim tensor to integer

        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features = 2048, out_features=1024)
        self.full_conn4 = nn.Linear(in_features = 1024, out_features = 2)
        #self.full_conn3 = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches

        x = self.full_conn1(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.relu(x)

        x = self.full_conn3(x)
        x = F.relu(x)

        x = self.full_conn4(x)

        return x

class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=108, out_channels=96, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=2),
            #nn.BatchNorm2d(num_features=96),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
            #nn.LocalResponseNorm(size=2),
            #nn.BatchNorm2d(num_features=256),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,  stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        mock_input = torch.randn(32, 108, 224, 224)
        mock_output = self.model(mock_input)
        flattened_output = torch.flatten(mock_output, start_dim=1)
        fc_in_dim = flattened_output.shape[1] # Get number of nodes from flattened value's size, then convert 0 dim tensor to integer

        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        self.norm1 = nn.BatchNorm1d(num_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)
        #self.full_conn3 = nn.Linear(in_features=4096, out_features=2)


    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches

        x = self.full_conn1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.relu(x)
        x = F.dropout(x)

        x = self.full_conn3(x)

        return x
