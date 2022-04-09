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
# Model
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

data_file
# Datasets and loaders
dataset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split=data_split,
    scale_mode=ckpt['args'].scale_mode
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

all_encodings = []

for i, batch in enumerate(tqdm(dataloader)):
    ref = batch['pointcloud'].to(args.device)

    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        all_encodings.append(code)

all_encodings = torch.cat(all_encodings, dim=0)

np.save(os.path.join(args.save_dir, 'cached_embeddings_data.npy'), all_encodings.cpu().numpy())
