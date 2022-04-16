import os
import time
import argparse
import torch
from tqdm.auto import tqdm
import h5py
from pathlib import Path
from torch.utils.data import DataLoader

from diffusion_point_cloud.models.autoencoder import *
from diffusion_point_cloud.utils.misc import *
from utilities.paths import PRETRAINED_FOLDER, DATA_FOLDER


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=str(PRETRAINED_FOLDER / 'AE_all.pt'))
parser.add_argument('--save_path', type=str, default=str(DATA_FOLDER / "aligned_embeddings_data.hdf5"))
parser.add_argument('--device', '-d', type=str, default='cpu')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=str(DATA_FOLDER / 'small_data_aligned_pc_data.hdf5'))
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_split', type=str, default='train', help='The data fold you want to cache (training, test, validation)')
args = parser.parse_args()

data_split = args.data_split


# Checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(args.ckpt)
else:
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))

seed_all(ckpt['args'].seed)

# Datasets and and output file
input_file = h5py.File(args.dataset_path)
output_file = h5py.File(args.save_path, "w")

# Model
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_encodings = []
for cate in input_file.keys():
    output_file.create_group(cate)
    for split in input_file[cate].keys():
        dataloader = DataLoader(input_file[cate][split][()], batch_size=args.batch_size, num_workers=0)
        embeddings = None
        for i, batch in enumerate(tqdm(dataloader)):
            ref = batch.to(args.device)
            model.eval()
            with torch.no_grad():
                if isinstance(embeddings, np.ndarray):
                    embeddings = np.append(embeddings, model.encode(ref).cpu().detach().numpy(), axis = 0)
                else:
                    if 'GEN' in args.ckpt:
                        embeddings = model.encode(ref).cpu().detach().numpy()
                    else:
                        embeddings = model.encode(ref).cpu().detach().numpy()
        if isinstance(embeddings, np.ndarray):
            output_file[cate].create_dataset(split, data=embeddings)
