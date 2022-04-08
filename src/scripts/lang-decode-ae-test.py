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
from langencoders.encoders.t5 import T5Enconder



# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=str(DIFFUSION_MODEL_PRETRAINED_FOLDER / 'AE_all.pt'))
parser.add_argument('--categories', type=str_list, default=['table'])
parser.add_argument('--save_dir', type=str, default=str(DIFFUSION_MODEL_RESULTS_FOLDER))
parser.add_argument('--device', '-d', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=str(DIFFUSION_MODEL_DATA_FOLDER / 'aligned_pc_data.hdf5'))
parser.add_argument('--lang_dataset_path', type=str, default=str(DIFFUSION_MODEL_DATA_FOLDER / 'aligned_text_data.hdf5'))

parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

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
    lang_path=args.lang_dataset_path,
    cates=args.categories,
    split='train',
    scale_mode=ckpt['args'].scale_mode
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

# Model
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

lang_model = T5Enconder()
lang_model.load_state_dict(torch.load(DIFFUSION_MODEL_PRETRAINED_FOLDER / 'language_test.pt'))
lang_model.to(args.device)

all_ref = []
all_recons = []
all_lang_recons = []

for i, batch in enumerate(tqdm(dataloader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    descriptions = batch['description']
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        lang_code = lang_model.encode(descriptions)
        recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
        lang_recons = model.decode(lang_code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref = ref * scale + shift
    recons = recons * scale + shift
    lang_recons = lang_recons * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())
    all_lang_recons.append(lang_recons.detach().cpu())

    break

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)
all_lang_recons = torch.cat(all_lang_recons, dim=0)

np.save(os.path.join(args.save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(args.save_dir, 'out.npy'), all_recons.numpy())
np.save(os.path.join(args.save_dir, 'lang_out.npy'), all_lang_recons.numpy())


