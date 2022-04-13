from typing import List
import torch.nn.functional as F
from torch import nn
import torch


class SimpleEncoder(nn.Module):
    """
    A simple encoder that will take the output of some language model backbone and put it in the correct size for the
    diffusion models.

    There's not a lot of learning happening in this layer. This is fine for large backbones that are being finetuned,
    however, it might not work well for small or frozen models where most param updates are expected to happen in the
    new encoder layer.
    """

    latent_dim: int

    def __init__(self, latent_dim: int = 256, *args, **kwargs):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Conv1d(1, self.latent_dim, (1,))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)
        x = self.conv(x)
        x = F.avg_pool1d(x, 1)
        x = F.max_pool1d(x, x.shape[-1]).squeeze(-1)
        return x
