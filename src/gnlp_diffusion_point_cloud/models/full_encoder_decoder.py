from torch import nn
from torch.nn import Module
from torch import optim
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, ClassVar

from diffusion_point_cloud.models.diffusion import DiffusionPoint
from diffusion_point_cloud.models.encoders.pointnet import PointNetEncoder
from models.backbones.backbone import Backbone
from models.backbones.t5 import T5Backbone
from models.backbones.simple import SimpleBackbone
from models.encoders.cnn_to_ff import CNNToFF
from models.encoders.simple_encoder import SimpleEncoder
from models.losses.contrastive_loss import ContrastiveLoss
from diffusion_point_cloud.models.diffusion import *


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


class FullEncoderDecoder(nn.Module):
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
            args
    ):
        super().__init__()
        self.args = args
        self.backbone_type = args.backbone
        self.encoder_type = args.encoder
        self.pc_encoder = PointNetEncoder(zdim=args.latent_dim)
        self.backbone = __back_bones__[self.backbone_type](args.size)
        for param in self.backbone.parameters():
            param.requires_grad = not args.backbone_freeze
        self.text_encoder = __encoders__[self.encoder_type](latent_dim=args.latent_dim)
        self.decoder = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        self.contrast_loss = ContrastiveLoss(negative_temperature = args.contrast_temp)
        self.l1_loss = torch.nn.SmoothL1Loss()

    def encode_text(self, tokens):
        return self.text_encoder(self.backbone(tokens))
    def encode_pc(self, pc):
        return self.pc_encoder(pc)[0]
    def decode(self, code, points):
        return self.decoder.sample(points, code)
    def get_loss(self, tokens, pc, weight = (1.0, 1.0, 1.0, 0.0), return_components = True):
        code_text = self.encode_text(tokens)
        code_pc = self.encode_pc(pc)
        l1_loss = self.l1_loss(code_text, code_pc)
        contrast_loss = self.contrast_loss(code_text, code_pc)
        text_loss = self.decoder.get_loss(pc, code_text)
        pc_loss = self.decoder.get_loss(pc, code_pc)
        sum_loss = weight[0] * contrast_loss + weight[1] * text_loss + weight[2] * pc_loss + weight[3] * l1_loss
        if (return_components):
            return sum_loss, contrast_loss, text_loss, pc_loss, l1_loss
        else:
            return sum_loss
