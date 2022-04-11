from typing import List
import torch.nn.functional as F
from torch import nn
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

T5Sizes ={
    "small": 'google/t5-v1_1-small',
    "base": 'google/t5-v1_1-base',
    "large": 'google/t5-v1_1-large',
    "extra_large": 'google/t5-v1_1-xl',
    "largest": 'google/t5-v1_1-xxl'
    }


class T5Encoder(nn.Module):

    def __init__(self,args):
        super().__init__()
        self.size = T5Sizes[args.size]
        self.encoding_size = args.latent_dim

        self.tokenizer = T5Tokenizer.from_pretrained(self.size)
        self.tokenizer.padding_size = "left"
        self.tokenizer.truncation_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = T5ForConditionalGeneration.from_pretrained(self.size)
        self.c1_m = nn.Conv1d(args.token_length, 256, 1)
        self.c2_m = nn.Conv1d(256, 256, 1)
        self.c3_m = nn.Conv1d(256, 512, 1)
        self.c4_m = nn.Conv1d(512, 1024, 1)
        self.c_bn1_m = nn.BatchNorm1d(256)
        self.c_bn2_m = nn.BatchNorm1d(256)
        self.c_bn3_m = nn.BatchNorm1d(512)
        self.c_bn4_m = nn.BatchNorm1d(1024)

        self.fc1_m = nn.Linear(1024, 512)
        self.fc2_m = nn.Linear(512, 256)
        self.fc3_m = nn.Linear(256, self.encoding_size)
        self.fc_bn1_m = nn.BatchNorm1d(512)
        self.fc_bn2_m = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.model.encoder(input_ids=x)[0]
        x = F.relu(self.c_bn1_m(self.c1_m(x)))
        x = F.relu(self.c_bn2_m(self.c2_m(x)))
        x = F.relu(self.c_bn3_m(self.c3_m(x)))
        x = F.relu(self.c_bn4_m(self.c4_m(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        x = F.relu(self.fc_bn2_m(self.fc2_m(x)))
        x = self.fc3_m(x)

        return x
