from torch import nn
from typing import List


class LanguageEncoder(nn.Module):
    """
    Base class for any LanguageEncoder
    """

    def encode(self, descriptions: List[str]):
        raise NotImplementedError("Implement the encode function for all LanguageEncoders")

    def train(self):
        pass