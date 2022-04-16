from torch import nn
from torch.nn import Module
from torch import optim
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, ClassVar

from models.backbones.backbone import Backbone
from models.backbones.t5 import T5Backbone
from models.backbones.simple import SimpleBackbone
from models.encoders.cnn_to_ff import CNNToFF
from models.encoders.simple_encoder import SimpleEncoder
from models.losses.contrastive_loss import ContrastiveLoss
from diffusion_point_cloud.models.diffusion import *
from utils.dataset import NLShapeNetCoreEmbeddings


# Helpful containers of backbones, encoders, and losses that we can easily look up in code.
# Dunders (__) so you don't accidentally import them else where, they are only really useful here.
__back_bones__: Dict[str, 'ClassVar[Backbone]'] = {
    'T5': T5Backbone,
    'SIMPLE': SimpleBackbone
}

__encoders__: Dict[str, 'ClassVar[Module]'] = {
    'SIMPLE': SimpleEncoder,
    'CNN2FF': CNNToFF
}

__losses__: Dict[str, 'ClassVar[Module]'] = {
    'MSE': nn.MSELoss,
    'ContrastiveCos': nn.CosineEmbeddingLoss,
    'ContrastiveLoss': ContrastiveLoss,
    'DiffusionMSE': DiffusionPoint
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
        for param in self.backbone.parameters():
            param.requires_grad = not kwargs["backbone_freeze"]
        self.encoder = __encoders__[self.encoder_type](*args, **kwargs)

        if self.loss_type == 'ContrastiveLoss':
            self.loss = __losses__[self.loss_type](*args, **kwargs)
        elif self.loss_type == "DiffusionMSE":
            self.loss = __losses__[self.loss_type](
                net = PointwiseNet(point_dim=3, context_dim=kwargs["latent_dim"], residual=kwargs["residual"]),
                var_sched = VarianceSchedule(
                    num_steps=kwargs["num_steps"],
                    beta_1=kwargs["beta_1"],
                    beta_T=kwargs["beta_T"],
                    mode=kwargs["sched_mode"]
                )
            )
        elif self.loss_type == "DiffusionGenMSE":
            pass  # This loss is specific to the generation model used during validation/cached embeddings curation.
        else:
            self.loss = __losses__[self.loss_type]()

    def encode(self, x):
        return self.encoder(self.backbone(x))

    def get_loss(self, x, y, *args, pcs=None, targets=None, gen_model=None, **kwargs):
        if self.loss_type == "ContrastiveLoss":
            return self.loss(self.encode(x), y, *args, **kwargs)
        if self.loss_type == "ContrastiveCos":
            return self.loss(self.encode(x), y, targets)
        if self.loss_type == "DiffusionMSE":
            return self.loss.get_loss(pcs, self.encode(x))
        if self.loss_type == "DiffusionGenMSE":
            return gen_model.diffusion.get_loss(pcs, self.encode(x))
        else:
            return self.loss(self.encode(x), y)
