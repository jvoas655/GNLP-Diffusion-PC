from typing import List
import torch.nn.functional as F
from torch import nn
import torch


class FFToCNN(nn.Module):
    """
    TODO - is this feasible? There are a lot of parameters here.
    TODO - torch.max seems incorrect
    TODO - x.view(-1, 1024) seems incorrect -- x.view(batch_size, -1) seems right, currently you could mix examples!
    TODO - so many batchnorms for language?
    """

    def __init__(self, latent_dim: int = 256, token_length: int = 128, *args, **kwargs):
        super().__init__()

        self.embedding_size = latent_dim

        self.fc1_m = nn.Linear(256, self.embedding_size)
        self.fc2_m = nn.Linear(256, 512)
        self.fc3_m = nn.Linear(512, 512)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(512)
        self.fc_bn3_m = nn.BatchNorm1d(512)

        self.ct1_m = nn.ConvTranspose1d(1, 16, 1)
        self.ct2_m = nn.ConvTranspose1d(16, 64, 1)
        self.ct3_m = nn.ConvTranspose1d(64, token_length, 1)
        self.ct_bn1_m = nn.BatchNorm1d(16)
        self.ct_bn2_m = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        x = F.relu(self.fc_bn2_m(self.fc2_m(x)))
        x = F.relu(self.fc_bn3_m(self.fc3_m(x)))
        x = x.view(-1, 1, 512)
        x = F.relu(self.ct_bn1_m(self.ct1_m(x)))
        x = F.relu(self.ct_bn2_m(self.ct2_m(x)))
        x = self.ct3_m(x)
        return x
