from typing import List
import torch.nn.functional as F
from torch import nn
import torch

class GraphEncoder(nn.Module):

    def __init__(self, latent_dim: int = 256, *args, **kwargs):
        super().__init__()

        self.embedding_size = latent_dim

        self.fc1_m = nn.Linear(12, 12)
        self.fc2_m = nn.Linear(12, 32)
        self.fc3_m = nn.Linear(32, 64)
        self.fc_bn1_m = nn.LayerNorm(12)
        self.fc_bn2_m = nn.LayerNorm(32)
        self.fc_bn3_m = nn.LayerNorm(64)

        self.gcn1_m = nn.Conv1d(64, 256, 1)
        self.gcn2_m = nn.Conv1d(256, 1024, 1)
        self.gcn_bn1_m = nn.BatchNorm1d(256)
        self.gcn_bn2_m = nn.BatchNorm1d(1024)

        self.max_pool_m = nn.MaxPool2d((2048, 1))

        self.fc4_m = nn.Linear(1024, self.embedding_size)
        self.fc5_m = nn.Linear(self.embedding_size, self.embedding_size)
        self.fc_bn4_m = nn.BatchNorm1d(self.embedding_size)
    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        x = F.relu(self.fc_bn2_m(self.fc2_m(x)))
        x = F.relu(self.fc_bn3_m(self.fc3_m(x)))
        x = F.relu(self.gcn_bn1_m(self.gcn1_m(x.transpose(2,1))))
        x = F.relu(self.gcn_bn2_m(self.gcn2_m(x)))
        x = self.max_pool_m(x.transpose(2,1))
        x = x.view(-1, 1024)
        x = F.relu(self.fc_bn4_m(self.fc4_m(x)))
        x = self.fc5_m(x)
        return x
