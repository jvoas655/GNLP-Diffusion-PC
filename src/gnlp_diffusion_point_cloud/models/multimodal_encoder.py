from torch import nn
from torch.nn import Module
from torch import optim

class MultiModalEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def encode_text(self, x, *args, **kwargs):
        return NotImplementedError
    def encode_pc(self, x, *args, **kwargs):
        return NotImplementedError
    def decode_text(self, code, *args, **kwargs):
        return NotImplementedError
    def decode_pc(self, code, *args, **kwargs):
        return NotImplementedError
    def get_loss(self, *args, **kwargs):
        return NotImplementedError
