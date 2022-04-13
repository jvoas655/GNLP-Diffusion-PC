from torch import nn
from torch.nn import Module
from torch import optim
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, ClassVar

from models.backbones.backbone import Backbone
from models.backbones.t5 import T5Backbone
from models.encoders.cnn_to_ff import CNNToFF
from models.encoders.simple_encoder import SimpleEncoder
from models.losses.contrastive_loss import ContrastiveLoss
from utils.dataset import NLShapeNetCoreEmbeddings


# Helpful containers of backbones, encoders, and losses that we can easily look up in code.
# Dunders (__) so you don't accidentally import them else where, they are only really useful here.
__back_bones__: Dict[str, 'ClassVar[Backbone]'] = {
    'T5': T5Backbone
}

__encoders__: Dict[str, 'ClassVar[Module]'] = {
    'SIMPLE': SimpleEncoder,
    'CNN2FF': CNNToFF
}

__losses__: Dict[str, 'ClassVar[Module]'] = {
    'MSE': nn.MSELoss,
    'ContrastiveCos': nn.CosineEmbeddingLoss,
    'ContrastiveLoss': ContrastiveLoss
}


class LanguageEncoder(nn.Module):
    """
    The langauge encoding model

    This class handles the raw input, language encoding, encoding to point cloud embeddings, and generating loss.
    """

    backbone: Backbone
    encoder: nn.Module
    loss: nn.Module

    backbone_type: str
    encoder_type: str
    loss_type: str


    def __init__(
            self,
            backbone: str = "T5",
            encoder: str = "CNN2FF",
            loss: str = "MSE",
            *args,
            **kwargs
    ):
        super().__init__()
        self.backbone_type = backbone
        self.encoder_type = encoder
        self.loss_type = loss

        self.backbone = __back_bones__[self.backbone_type](*args, **kwargs)
        self.encoder = __encoders__[self.encoder_type](*args, **kwargs)

        if self.loss_type == 'ContrastiveLoss':
            self.loss = __losses__[self.loss_type](*args, **kwargs)
        else:
            self.loss = __losses__[self.loss_type]()

    def encode(self, x):
        return self.encoder(self.backbone(x))

    def get_loss(self, x, y, *args, targets=None, **kwargs):
        if self.loss_type == "ContrastiveLoss":
            return self.loss(self.encode(x), y, *args, **kwargs)
        if self.loss_type == "ContrastiveCos":
            return self.loss(self.encode(x), y, targets)
        else:
            return self.loss(self.encode(x), y)
