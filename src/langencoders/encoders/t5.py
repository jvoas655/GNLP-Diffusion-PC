from langencoders.langauge_encoder import LanguageEncoder
from langencoders.utils.vocabulary import Vocabulary, Tokenizer
from typing import List

import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Sizes:

    small = 'google/t5-v1_1-small'
    base = 'google/t5-v1_1-base'
    large = 'google/t5-v1_1-large'
    extra_large = 'google/t5-v1_1-xl'
    largest = 'google/t5-v1_1-xxl'


class T5Enconder(LanguageEncoder):

    def __init__(
            self,
            size=T5Sizes.small

    ):
        super().__init__()
        self.size = size

        self.tokenizer = T5Tokenizer.from_pretrained(self.size)
        self.model = T5ForConditionalGeneration.from_pretrained(self.size)

        self.C1 = torch.nn.Conv1d(1, 256, 1)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        hidden = self.model.encoder(input_ids=x)[0]

        # TODO - we almost definitely want to do more things to this encoding than reduce the dimensions right?
        map = self.C1(hidden.view(batch_size, -1))

        embedding = torch.nn.functional.avg_pool1d(map, map.shape[-1]).squeeze(-1)
        embedding = embedding.view(batch_size, -1)
        return embedding

    def encode(self, descriptions: List[str]):
        task_prefix = "Create a pointcloud described as: "

        input_ids = self.tokenizer(
            [task_prefix + desc for desc in descriptions],
            padding=True,
            return_tensors="pt"
        ).input_ids

        return self.forward(input_ids)


if __name__ == "__main__":
    loss = torch.nn.CrossEntropyLoss()


    desc = "A chair with three legs."

    model = T5Enconder()

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    encoding = model.encode([desc])
    dupe_encoding = model.encode([desc])

    test_vector = torch.ones([1, 256])

    output = loss(encoding, test_vector)
    output.backward()
    optim.step()
    optim.zero_grad()

    updated_encoding = model.encode([desc])


    print(encoding)