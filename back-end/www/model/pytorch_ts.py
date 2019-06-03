import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=96, kernel_size=7, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv3d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1),
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        mock_input = torch.randn(4, 2, 36, 224, 224)
        mock_output = self.model(mock_input)
        flattened_outputed = torch.flatten(mock_output, start_dim=1)
        fc_in_dim = flattened_outputed[1]

        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches

        x = self.full_conn1(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.dropout(x)

        x = self.full_conn3(x)

        return F.softmax(x, dim=0)

class SpatialCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=96, kernel_size=7, stride=2),
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv3d(in_channels=96, out_channels=256, kernel_size=5,
                      stride=1),
            nn.MaxPool3d(kernel_size=3, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),

            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3,
                      stride=1),
            nn.ReLU(),
        )

        mock_input = torch.randn(4, 3, 36, 224, 224)
        mock_output = self.model(mock_input)
        flattened_outputed = torch.flatten(mock_output, start_dim=1)
        fc_in_dim = flattened_outputed[1]

        self.full_conn1 = nn.Linear(in_features=fc_in_dim, out_features=4096)
        self.full_conn2 = nn.Linear(in_features=4096, out_features=2048)
        self.full_conn3 = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x, start_dim=1)  # Flattens layers without losing batches

        x = self.full_conn1(x)
        x = F.dropout(x)

        x = self.full_conn2(x)
        x = F.dropout(x)

        x = self.full_conn3(x)

        return F.softmax(x, dim=0)