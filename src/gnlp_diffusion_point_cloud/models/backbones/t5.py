from typing import List, Union

import torch.nn.functional as F
from torch import nn
import torch
from enum import Enum

from transformers import T5Tokenizer, T5ForConditionalGeneration

from models.backbones.backbone import Backbone


class T5Size(Enum):
    """
    This is an enumeration class, used for text completion when coding to help remember long strings of weird values.
    """

    small = 'google/t5-v1_1-small'
    base = 'google/t5-v1_1-base'
    large = 'google/t5-v1_1-large'
    extra_large = 'google/t5-v1_1-xl'
    largest = 'google/t5-v1_1-xxl'


class T5Backbone(Backbone):

    size: str
    token_length: int

    def __init__(
            self,
            size: Union[str, T5Size],
            *args,
            token_length: int = 128,
            **kwargs
    ):
        """
        T5 is a Decoder-Only large language transformer.
        Paper: https://arxiv.org/pdf/1910.10683.pdf
        Hugging face: https://huggingface.co/docs/transformers/model_doc/t5

        It is trained on various tasks via prompts
        'Translate this sentence from english to german'
        etc.

        The idea behind using it here is that it might have good priors on how to describe certain objects.  I.E.
        chairs and tables have "legs" which would be represented close in the embedding space and far away from
        other descriptions like "tires".

        We could then utilize this pretrained embedding space that already differentiates these descriptions to
        finetune onto the point cloud embedding space.
        """
        super().__init__()
        if isinstance(size, str):
            self.size = T5Size[size].value
        else:
            self.size = size.value

        self.token_length = token_length

        # Base tokenizer, we will wrap it with our own function to handle special inputs
        # and setting task inputs. Technically, you could just call this directly though that's
        # usually a bad idea because you have parameters here that other classes/funcs don't need
        # to know about.
        self.__tokenizer__ = T5Tokenizer.from_pretrained(self.size)
        self.__tokenizer__.padding_size = "left"
        self.__tokenizer__.truncation_side = "left"
        self.__tokenizer__.pad_token = self.__tokenizer__.eos_token

        self.model = T5ForConditionalGeneration.from_pretrained(self.size)

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

    def encode(self, x):
        return self.model.encoder(input_ids=x)[0]
