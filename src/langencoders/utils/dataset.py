from typing import List, Callable
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path

from utilities.paths import DIFFUSION_MODEL_DATA_FOLDER


class NaturalLanguageShapeNetCore(Dataset):
    descriptions: List[str]
    cached_embeddings: List[np.array]
    transform: Callable

    def __init__(
            self,
            descriptions: List[List[str]],
            cached_embeddings: List[List[np.array]],
            transform: Callable = None
    ):
        super().__init__()

        self.descriptions = []
        [self.descriptions.extend(x) for x in descriptions]
        self.embeddings = []
        [self.embeddings.extend(x) for x in cached_embeddings]

        assert len(self.descriptions) == len(self.embeddings), 'number of descriptions does NOT match number of pcs.'

        self.transform = transform

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        description = self.descriptions[idx]
        embedding = self.embeddings[idx]

        if self.transform:
            description = self.transform(description)

        return description, embedding

    @classmethod
    def from_h5_file(
            cls,
            text_data_file: Path = DIFFUSION_MODEL_DATA_FOLDER / 'aligned_text_data.hdf5',
            cached_embeddings_file: Path = DIFFUSION_MODEL_DATA_FOLDER / 'cached_embeddings_data.npy',
            text_category_ids: List[str] = ('04379243'),
            data_fold: str = 'train'
    ):
        # TODO - get cached embeddings.
        text_data_file = h5py.File(str(text_data_file), 'r')
        text_data = text_data_file[text_category_ids][data_fold]

        point_clouds = [np.load(str(cached_embeddings_file))]

        return NaturalLanguageShapeNetCore(text_data, point_clouds)

if __name__ == "__main__":
    dataset = NaturalLanguageShapeNetCore.from_h5_file()