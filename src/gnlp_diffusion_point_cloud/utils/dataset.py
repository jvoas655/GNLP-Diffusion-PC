from typing import List, Callable
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from pathlib import Path

from utilities.paths import DATA_FOLDER

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class NLShapeNetCoreEmbeddings(Dataset):

    def __init__(
            self,
            desc_path,
            embedding_path,
            tokenizer,
            token_length,
            cates,
            split,
            transform: Callable = None,
            size: int = -1,
    ):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        self.token_length = token_length
        self.desc_path = desc_path
        self.embedding_path = embedding_path
        self.tokenizer = tokenizer
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.transform = transform
        self.stats = None

        # The size of the dataset, -1 means the entire dataset, if you want to run experiments on small sets for
        # convergence testing etc. set this to a low value.
        self.size = size

        self.descriptions = []
        self.embeddings = []
        self.token_vectors = []
        self.get_statistics()
        self.load()

        assert len(self.descriptions) == len(self.embeddings), 'number of descriptions does NOT match number of pcs.'

        self.transform = transform
    def get_statistics(self):
        return

    def load(self):
        text_data_file = h5py.File(self.desc_path)
        cached_embeddings_file = h5py.File(self.embedding_path)
        encoding = 'utf-8'
        self.embeddings = None
        for synsetid in self.cate_synsetids:
            for j, desc in enumerate(text_data_file[synsetid][self.split]):
                self.descriptions.append(desc.decode(encoding))
            if (isinstance(self.embeddings, np.ndarray)):
                self.embeddings = np.append(self.embeddings, cached_embeddings_file[synsetid][self.split][()], axis=0)
            else:
                self.embeddings = cached_embeddings_file[synsetid][self.split][()]

        self.token_vectors = self.tokenizer(self.descriptions)
        self.descriptions = np.array(self.descriptions)

        # TODO - why is this here? This should be somewhere very clear at the top of a script, this will lead to
        # bugs when you want to make something random but are confused on why it's getting the same results.
        np.random.seed(2022)
        shuffle_inds = np.random.shuffle(np.arange(self.descriptions.shape[0]))

        self.token_vectors = np.squeeze(self.token_vectors.detach().numpy()[shuffle_inds, :], axis=0)
        self.descriptions = np.squeeze(self.descriptions[shuffle_inds], axis=0)
        self.embeddings = np.squeeze(self.embeddings[shuffle_inds, :], axis=0)

        # Restrict the size of the dataset to the size parameter if it was supplied.
        if self.size > 0:
            # Max out the size of the dataset to whatever the actual size of the dataset is
            # (i.e. you can't size a size of 1 million with a dataset of thousands)
            self.token_vectors = self.token_vectors[0:min(len(self.token_vectors), self.size)]
            self.descriptions = self.descriptions[0:min(len(self.descriptions), self.size)]
            self.embeddings = self.embeddings[0:min(len(self.embeddings), self.size)]

    def __len__(self):
        return self.descriptions.shape[0]

    def __getitem__(self, idx):
        return self.token_vectors[idx, :], self.embeddings[idx, :]

class NLShapeNetCoreJoint(Dataset):
    pass # Will be a join contrastive learning algo
