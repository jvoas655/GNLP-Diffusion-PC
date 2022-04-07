from langencoders.langauge_encoder import LanguageEncoder
from langencoders.utils.vocabulary import Vocabulary, Tokenizer
from typing import List

import torch


class ToyLanguageEncoder(LanguageEncoder):

    def __init__(
            self,
            tokenizer: Tokenizer,
            embedding_size: int = 8,
            latent_dim: int = 256

    ):
        super().__init__()

        self.tokenizer = tokenizer

        self.E = torch.nn.Embedding(num_embeddings=len(self.tokenizer.vocabulary), embedding_dim=embedding_size, padding_idx=0)
        self.P = torch.nn.Linear(embedding_size, latent_dim)

    def forward(self, x: torch.Tensor):
        embeddings = self.E(x)
        projections = self.P(embeddings)
        return projections

    def encode(self, descriptions: List[str]):
        tokens = [self.tokenizer.convert_tokens_to_ids(x.split(" ")) for x in descriptions]
        x = torch.LongTensor(tokens)
        return self(x)


if __name__ == "__main__":
    desc = "A chair with three legs."

    vocab = Vocabulary.create_word_vocab([desc.split(" ")], 100)
    tokenizer = Tokenizer(vocab)

    model = ToyLanguageEncoder(tokenizer)
    encoding = model.encode([desc])
    print(encoding)