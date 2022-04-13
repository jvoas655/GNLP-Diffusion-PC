from typing import List
import torch.nn.functional as F
from torch import nn
import torch


class CNNToFF(nn.Module):
    """
    TODO - is this feasible? There are a lot of parameters here.
    TODO - torch.max seems incorrect
    TODO - x.view(-1, 1024) seems incorrect -- x.view(batch_size, -1) seems right, currently you could mix examples!
    TODO - so many batchnorms for language?
    """

    def __init__(self, latent_dim: int = 256, token_length: int = 128, *args, **kwargs):
        super().__init__()

        self.embedding_size = latent_dim

        self.c1_m = nn.Conv1d(token_length, 256, 1)
        self.c2_m = nn.Conv1d(256, 256, 1)
        self.c3_m = nn.Conv1d(256, 512, 1)
        self.c4_m = nn.Conv1d(512, 1024, 1)
        self.c_bn1_m = nn.BatchNorm1d(256)
        self.c_bn2_m = nn.BatchNorm1d(256)
        self.c_bn3_m = nn.BatchNorm1d(512)
        self.c_bn4_m = nn.BatchNorm1d(1024)

        self.fc1_m = nn.Linear(1024, 512)
        self.fc2_m = nn.Linear(512, 256)
        self.fc3_m = nn.Linear(256, self.embedding_size)
        self.fc_bn1_m = nn.BatchNorm1d(512)
        self.fc_bn2_m = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = F.relu(self.c_bn1_m(self.c1_m(x)))
        x = F.relu(self.c_bn2_m(self.c2_m(x)))
        x = F.relu(self.c_bn3_m(self.c3_m(x)))
        x = F.relu(self.c_bn4_m(self.c4_m(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        x = F.relu(self.fc_bn2_m(self.fc2_m(x)))
        x = self.fc3_m(x)
        return x