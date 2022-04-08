from torch import nn
from torch import optim
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm

from langencoders.utils.dataset import NaturalLanguageShapeNetCore


class LanguageEncoder(nn.Module):
    """
    Base class for any LanguageEncoder
    """

    def encode(self, descriptions: List[str]):
        raise NotImplementedError("Implement the encode function for all LanguageEncoders")

    def train_epoch(
            self,
            dataloader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer
    ):
        for idx, (descriptions, embeddings) in tqdm(enumerate(dataloader), desc='Training Epoch', total=len(dataloader), position=1, leave=False):
            optimizer.zero_grad()

            encodings = self.encode(descriptions)

            loss = criterion(embeddings, encodings)
            loss.backward()

            optimizer.step()


if __name__ == "__main__":
    from langencoders.encoders.t5 import T5Enconder
    from utilities.paths import DIFFUSION_MODEL_PRETRAINED_FOLDER
    import numpy as np
    import torch

    dataset = NaturalLanguageShapeNetCore.from_h5_file(max_size=2)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0)
    criterion = nn.MSELoss()

    model = T5Enconder()
    optimizer = optim.Adam(model.parameters(recurse=True), lr=0.001)

    model.train()
    for i in range(5):
        model.train_epoch(dataloader, criterion=criterion, optimizer=optimizer)

    torch.save(model.state_dict(), str(DIFFUSION_MODEL_PRETRAINED_FOLDER / 'language_test.pt'))



