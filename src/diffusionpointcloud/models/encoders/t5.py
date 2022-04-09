from ..tokenizers.vocabulary import Vocabulary, Tokenizer
from typing import List
import torch.nn.functional as F
from torch import nn

from transformers import T5Tokenizer, T5ForConditionalGeneration

T5Sizes ={
    "small": 'google/t5-v1_1-small',
    "base": 'google/t5-v1_1-base',
    "large": 'google/t5-v1_1-large',
    "extra_large": 'google/t5-v1_1-xl',
    "largest": 'google/t5-v1_1-xxl'
    }


class T5Enconder(nn.Module):

    def __init__(self,args):
        super().__init__()
        self.size = T5Sizes[args.size]
        self.encoding_szie = args.latent_dim

        self.tokenizer = T5Tokenizer.from_pretrained(self.size)
        self.model = T5ForConditionalGeneration.from_pretrained(self.size)

        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 256)
        self.fc3_m = nn.Linear(256, encoding_szie)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.model.encoder(input_ids=x)[0]

        # TODO - we almost definitely want to do more things to this encoding than reduce the dimensions right?
        x = f.relu(fc_bn1_m(fc1_m(x)))
        x = f.relu(fc_bn2_m(fc2_m(x)))
        x = fc3_m(x)

        return x

    def encode(self, descriptions: List[str]):
        task_prefix = "Create a pointcloud described as: "

        input_ids = self.tokenizer(
            [task_prefix + desc for desc in descriptions],
            padding=True,
            return_tensors="pt"
        ).input_ids

        return self.forward(input_ids)
