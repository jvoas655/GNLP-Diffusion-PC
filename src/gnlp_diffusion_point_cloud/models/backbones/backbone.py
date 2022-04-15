from torch.nn import Module


class Backbone(Module):
    """ Base class for backbone language models """

    def tokenizer(self, x):
        raise NotImplementedError("All backbones need to tokenizer input")

    def encode(self, x, *args):
        raise NotImplementedError("All backbones need to encode")

    def forward(self, x):
        x = self.encode(x)
        return x
