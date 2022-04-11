from torch import nn
from torch import optim
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoders.t5 import *
from utils.dataset import NLShapeNetCoreEmbeddings


class LanguageEncoder(nn.Module):
    """
    Base class for any LanguageEncoder
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        if (args.model == "T5"):
            self.encoder = T5Encoder(args)
        self.tokenizer = self.encoder.tokenizer

    def encode(self, x):
        return self.encoder(x)

    def get_loss(self, x, y):
        code = self.encoder(x)
        loss = nn.MSELoss()
        return loss(code, y)
