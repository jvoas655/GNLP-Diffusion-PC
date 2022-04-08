import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from diffusionpointcloud.utils.dataset import *
from diffusionpointcloud.utils.misc import *
from diffusionpointcloud.utils.data import *
from diffusionpointcloud.models.autoencoder import *
from diffusionpointcloud.evaluation import EMD_CD
from utilities.paths import DIFFUSION_MODEL_PRETRAINED_FOLDER, DIFFUSION_MODEL_RESULTS_FOLDER, DIFFUSION_MODEL_DATA_FOLDER

# TODO - make this selection of encoder an argument in the CLI
from langencoders.encoders.toy import ToyLanguageEncoder
from langencoders.utils.vocabulary import Tokenizer, Vocabulary


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=str(DIFFUSION_MODEL_PRETRAINED_FOLDER / 'AE_all.pt'))
parser.add_argument('--categories', type=str_list, default=['table'])
parser.add_argument('--save_dir', type=str, default=str(DIFFUSION_MODEL_RESULTS_FOLDER))
parser.add_argument('--device', '-d', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=str(DIFFUSION_MODEL_DATA_FOLDER / 'aligned_pc_data.hdf5'))
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_split', type=str, default='train', help='The data fold you want to cache (training, test, validation)')
args = parser.parse_args()

data_split = args.data_split

ckpt = DIFFUSION_MODEL_PRETRAINED_FOLDER / args.ckpt

# Checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(str(ckpt))
else:
    ckpt = torch.load(str(ckpt), map_location=torch.device('cpu'))
seed_all(ckpt['args'].seed)

# Datasets and loaders
dataset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split=data_split,
    scale_mode=ckpt['args'].scale_mode
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

# Model
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_encodings = []

for i, batch in enumerate(tqdm(dataloader)):
    ref = batch['pointcloud'].to(args.device)

    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        all_encodings.append(code)

all_encodings = torch.cat(all_encodings, dim=0)

np.save(os.path.join(args.save_dir, 'cached_embeddings_data.npy'), all_encodings.numpy())


