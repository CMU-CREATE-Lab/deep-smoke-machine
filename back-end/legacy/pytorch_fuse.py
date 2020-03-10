import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.full_conn1 = nn.Linear(in_features=4, out_features=256)
        self.full_conn4 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.full_conn1(x)

        x = F.relu(x)

        x = self.full_conn4(x)

        return F.softmax(x, dim=0)
