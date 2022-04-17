from typing import List
import torch.nn.functional as F
from torch import nn
import torch
from evaluation.evaluation_metrics import *

class FoldingNet(nn.Module):

    def __init__(self, latent_dim: int = 256, output_points = 2048, *args, **kwargs):
        super().__init__()

        self.embedding_size = latent_dim
        self.output_size = output_points

        self.fold_grid = torch.zeros((2, self.output_size))
        self.fold_grid[0, :] = torch.div(torch.linspace(0, 2 * self.output_size, steps=output_points), (self.output_size**0.5), rounding_mode = "trunc") / (self.output_size ** 0.5) - 1
        self.fold_grid[1, :] = 2 * torch.remainder(torch.linspace(0, self.output_size, steps=output_points), (self.output_size**0.5)) / (self.output_size ** 0.5) - 1
        self.fc1_m = nn.Linear(self.embedding_size + 2, 64)
        self.fc2_m = nn.Linear(64, 16)
        self.fc3_m = nn.Linear(16, 3)
        self.fc_bn1_m = nn.LayerNorm(64)
        self.fc_bn2_m = nn.LayerNorm(16)
        self.fc_bn3_m = nn.LayerNorm(3)

        self.fc4_m = nn.Linear(self.embedding_size + 3, 64)
        self.fc5_m = nn.Linear(64, 16)
        self.fc6_m = nn.Linear(16, 3)
        self.fc_bn4_m = nn.LayerNorm(64)
        self.fc_bn5_m = nn.LayerNorm(16)
    def to(self, device):
        super().to(device)
        self.fold_grid = self.fold_grid.to(device)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        batch_grid = self.fold_grid.view(1, self.output_size, 2).repeat((batch_size, 1, 1))
        embeddings = x.view(-1, 1, self.embedding_size).repeat((1, self.output_size, 1))
        x = torch.cat((embeddings, batch_grid), 2)
        x = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        x = F.relu(self.fc_bn2_m(self.fc2_m(x)))
        x = F.relu(self.fc_bn3_m(self.fc3_m(x)))
        x = torch.cat((embeddings, x), 2)
        x = F.relu(self.fc_bn4_m(self.fc4_m(x)))
        x = F.relu(self.fc_bn5_m(self.fc5_m(x)))
        x = self.fc6_m(x)
        return x
