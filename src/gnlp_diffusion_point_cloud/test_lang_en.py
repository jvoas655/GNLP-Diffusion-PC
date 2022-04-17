import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from diffusion_point_cloud.utils.dataset import *
from diffusion_point_cloud.utils.misc import *
from diffusion_point_cloud.utils.data import *
from diffusion_point_cloud.models.autoencoder import *
from diffusion_point_cloud.evaluation import EMD_CD
from utilities.paths import DATA_FOLDER
from utilities.paths import PRETRAINED_FOLDER

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=PRETRAINED_FOLDER / 'LEN_chair.pt')
parser.add_argument('--categories', type=str_list, default=["chair"])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--lang_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_text_data.hdf5")
parser.add_argument('--emb_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_embeddings_data.hdf5")
parser.add_argument('--pc_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_pc_data.hdf5")
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'LEN_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(args.ckpt)
else:
    ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()

    ref = ref * scale + shift
    recons = recons * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
logger.info('CD:  %.12f' % cd)
logger.info('EMD: %.12f' % emd)
