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
from models.encoders.graph_encoder import *
from models.multimodal_encoder import *
from models.encoders.cnn_to_ff import *
from models.decoders.ff_to_cnn import *
from models.decoders.folding_net import *
from evaluation.evaluation_metrics import *
from models.decoders.diffusion import *


# Helpful containers of backbones, encoders, and losses that we can easily look up in code.
# Dunders (__) so you don't accidentally import them else where, they are only really useful here.
__text_backbones__: Dict[str, 'ClassVar[Backbone]'] = {
    'T5': T5Backbone
}

__text_transfer__: Dict[str, 'ClassVar[Module]'] = {
    'SIMPLE': SimpleEncoder,
    'CNN2FF': CNNToFF
}

__pc_encoders__: Dict[str, 'ClassVar[Module]'] = {
    'GraphEncoder': GraphEncoder
}

__pc_decoders__: Dict[str, 'ClassVar[Module]'] = {
    'FoldingNet': FoldingNet
}

__text_decoders__: Dict[str, 'ClassVar[Module]'] = {
    'FF2CNN': FFToCNN
}


__losses__: Dict[str, 'ClassVar[Module]'] = {
    'MSE': nn.MSELoss,
    'ContrastiveCos': nn.CosineEmbeddingLoss,
}


class DCCAE(MultiModalEncoder):
    """
    The langauge encoding model

    This class handles the raw input, language encoding, encoding to point cloud embeddings, and generating loss.
    """
    def __init__(
            self,
            text_backbone: str = "T5",
            text_transfer: str = "CNN2FF",
            pc_encoder: str = "GraphEncoder",
            text_decoder: str = "FF2CNN",
            pc_decoder: str = "FoldingNet",
            loss: str = "MSE",
            *args,
            **kwargs
    ):
        super().__init__()
        self.loss_type = loss
        self.loss = __losses__[self.loss_type]()
        self.text_encoder_backbone_type = text_backbone
        self.text_encoder_backbone = __text_backbones__[self.text_encoder_backbone_type](*args, **kwargs)
        for param in self.text_encoder_backbone.parameters():
            param.requires_grad = not kwargs["backbone_freeze"]
        self.tokenizer = self.text_encoder_backbone.tokenizer
        self.text_encoder_transfer_type = text_transfer
        self.text_encoder_transfer = __text_transfer__[self.text_encoder_transfer_type](*args, **kwargs)
        self.pc_encoder_type = pc_encoder
        self.pc_encoder = __pc_encoders__[self.pc_encoder_type](kwargs["latent_dim"])
        self.text_decoder_type = text_decoder
        self.text_decoder = __text_decoders__[self.text_decoder_type](kwargs["latent_dim"])
        self.pc_decoder = self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=3, context_dim=kwargs["latent_dim"], residual=kwargs["residual"]),
            var_sched = VarianceSchedule(
                num_steps=kwargs["num_steps"],
                beta_1=kwargs["beta_1"],
                beta_T=kwargs["beta_T"],
                mode=kwargs["sched_mode"]
            )
        )

        return
        self.pc_decoder_type = pc_decoder
        self.pc_decoder = __pc_decoders__[self.pc_decoder_type](kwargs["latent_dim"])
    def to(self, device):
        super().to(device)
        self.text_encoder_backbone = self.text_encoder_backbone.to(device)
        self.text_encoder_transfer = self.text_encoder_transfer.to(device)
        self.text_decoder = self.text_decoder.to(device)
        self.pc_encoder = self.pc_encoder.to(device)
        self.pc_decoder = self.pc_decoder.to(device)
        return self
    def encode_pc(self, x, *args, **kwargs):
        return self.pc_encoder(x)
    def decode_pc(self, code, *args, **kwargs):
        return self.pc_decoder(code)
    def encode_text(self, x, *args, **kwargs):
        code1 = self.text_encoder_backbone(x)
        code2 = self.text_encoder_transfer(code1)
        return code1, code2
    def decode_text(self, code, *args, **kwargs):
        return self.text_decoder(code)
    def get_loss(self, *args, **kwargs):
        losses = {}
        if ("x_text" in kwargs):
            text_code1, text_code2 = self.encode_text(kwargs["x_text"])
            text_con = self.decode_text(text_code2)
            text_loss = self.loss(text_code1, text_con)
            if ("sum_loss" in losses):
                losses["sum_loss"] += text_loss
            else:
                losses["sum_loss"] = text_loss
            losses["text_loss"] = text_loss
        if ("x_pc" in kwargs):
            pc_code = self.encode_pc(kwargs["x_pc"])
            #pc_con = self.decode_pc(pc_code)
            #dl, dr = distChamfer(pc_con, kwargs["y_pc"])
            #pc_loss = torch.mean(dl.mean(dim=1) + dr.mean(dim=1))
            pc_loss = self.pc_decoder.get_loss(kwargs["y_pc"], pc_code)
            if ("sum_loss" in losses):
                losses["sum_loss"] += pc_loss
            else:
                losses["sum_loss"] = pc_loss
            losses["pc_loss"] = pc_loss
        if ("x_pc" in kwargs and "x_text" in kwargs):
            correlation_loss = self.loss(text_code2, pc_code)
            #dl, dr = distChamfer(self.decode_pc(text_code2), kwargs["y_pc"])
            #cross_text_loss = torch.mean(dl.mean(dim=1) + dr.mean(dim=1))
            cross_text_loss = self.pc_decoder.get_loss(kwargs["y_pc"], text_code2)
            if ("sum_loss" in losses):
                losses["sum_loss"] += correlation_loss + cross_text_loss
            else:
                losses["sum_loss"] = correlation_loss + cross_text_loss
            losses["correlation_loss"] = correlation_loss
            losses["cross_text_loss"] = cross_text_loss
        return losses
