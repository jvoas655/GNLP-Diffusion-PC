import os
import time
import argparse
import torch
from tqdm.auto import tqdm
from pathlib import Path

from diffusion_point_cloud.utils.dataset import *
from diffusion_point_cloud.utils.misc import *
from diffusion_point_cloud.utils.data import *
from diffusion_point_cloud.models.autoencoder import *
from diffusion_point_cloud.evaluation import EMD_CD
from utilities.paths import PRETRAINED_FOLDER, RESULTS_FOLDER, DIFFUSION_MODEL_FOLDER, DATA_FOLDER
from diffusion_point_cloud.models.vae_flow import *

# TODO - make this selection of encoder an argument in the CLI
from gnlp_diffusion_point_cloud.models.language_encoder import *


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=str(PRETRAINED_FOLDER / 'AE_all.pt'))
parser.add_argument('--lang_ckpt', type=str, required=True, help='Path to your language model pt file.')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--save_dir', type=str, default=str(RESULTS_FOLDER))
parser.add_argument('--device', '-d', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=str(DATA_FOLDER / 'aligned_pc_data.hdf5'))
parser.add_argument('--lang_dataset_path', type=str, default=str(DATA_FOLDER / 'aligned_text_data.hdf5'))
parser.add_argument('--emb_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_embeddings_data.hdf5")
parser.add_argument('--token_length', type=int, default=128)
parser.add_argument('--dataset_size', '-ds', type=int, default=-1,
                    help='-1 means all dataset examples are used, anything higher will slice the dataset.')

parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

ckpt = PRETRAINED_FOLDER / args.ckpt

lang_ckpt = torch.load(args.lang_ckpt)
lang_model = LanguageEncoder(**vars(lang_ckpt['args'])).to(args.device)
lang_model.load_state_dict(lang_ckpt['state_dict'])

# Checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(str(ckpt))
else:
    ckpt = torch.load(str(ckpt), map_location=torch.device('cpu'))
seed_all(ckpt['args'].seed)

# Datasets and loaders
dataset = NLShapeNetCoreEmbeddings(
    desc_path=args.lang_dataset_path,
    embedding_path=args.emb_dataset_path,
    pc_path=args.dataset_path,
    cates=args.categories,
    split='train',
    tokenizer=lang_model.backbone.tokenizer,
    token_length=args.token_length,
    transform=None,
    size=args.dataset_size,
    scale_mode='shape_unit'
)
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

# Model
# model = AutoEncoder(ckpt['args']).to(args.device)
model = FlowVAE(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_recons = []
all_lang_recons = []
all_refs = []

for i, batch in enumerate(tqdm(dataloader)):
    tokens, embeddings, pc = batch
    tokens = tokens.to(args.device)
    embeddings = embeddings.to(args.device)
    model.eval()
    with torch.no_grad():
        lang_code = lang_model.encode(tokens.to(args.device))
        recons = model.decode(embeddings, 2048, flexibility=ckpt['args'].flexibility).detach()
        lang_recons = model.decode(lang_code, 2048, flexibility=ckpt['args'].flexibility).detach()

    all_recons.append(recons.detach().cpu())
    all_lang_recons.append(lang_recons.detach().cpu())
    all_refs.append(pc.detach().cpu())

    if i == 32:
      break

all_recons = torch.cat(all_recons, dim=0)
all_lang_recons = torch.cat(all_lang_recons, dim=0)
all_refs = torch.cat(all_refs, dim=0)

np.save(os.path.join(args.save_dir, 'out.npy'), all_recons.numpy())
np.save(os.path.join(args.save_dir, 'lang_out.npy'), all_lang_recons.numpy())
np.save(os.path.join(args.save_dir, 'ref.npy'), all_refs.numpy())


