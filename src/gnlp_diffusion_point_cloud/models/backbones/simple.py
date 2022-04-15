import torch
from torch import nn
from torch.nn import functional as F
from typing import Union

from models.backbones.backbone import Backbone

from transformers import T5Tokenizer
from models.backbones.t5 import T5Size


class SimpleBackbone(Backbone):
    def __init__(self, size: Union[str, T5Size], *args, hidden_backbone_size=256, embedding_emb=128, token_length: int = 128, **kwargs):
        super(SimpleBackbone, self).__init__()
        if isinstance(size, str):
            self.size = T5Size[size].value
        else:
            self.size = size.value

        self.hidden_size = hidden_backbone_size
        self.token_length = token_length

        self.__tokenizer__ = T5Tokenizer.from_pretrained(self.size)
        self.__tokenizer__.padding_size = "left"
        self.__tokenizer__.truncation_side = "left"
        self.__tokenizer__.pad_token = self.__tokenizer__.eos_token

        self.embedding = nn.Embedding(len(self.__tokenizer__), self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.C = nn.Conv1d(1, 256, (1,))


    def encode(self, x, hidden=None):
        output = F.relu(self.C(F.relu(self.embedding(x)).view(x.shape[0], -1).unsqueeze(1)))
        output = F.max_pool1d(output, (output.shape[-1])).squeeze()
        output, _ = self.gru(output)

        output = self.softmax(self.out(output))
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def tokenizer(
            self,
            x,
            padding_type: str = "max_length",
            task_prefix: str = ""
    ):
        token_vectors = self.__tokenizer__(
            [task_prefix + desc for desc in x],
            padding=padding_type,
            max_length=self.token_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return token_vectors
