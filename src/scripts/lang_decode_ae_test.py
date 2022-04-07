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
parser.add_argument('--ckpt', type=str, default=str(DIFFUSION_MODEL_PRETRAINED_FOLDER / 'AE_airplane.pt'))
parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--save_dir', type=str, default=str(DIFFUSION_MODEL_RESULTS_FOLDER))
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=str(DIFFUSION_MODEL_DATA_FOLDER / 'shapenet.hdf5'))
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

ckpt = DIFFUSION_MODEL_PRETRAINED_FOLDER / args.ckpt

# Checkpoint
if torch.cuda.is_available():
    ckpt = torch.load(str(ckpt))
else:
    ckpt = torch.load(str(ckpt), map_location=torch.device('cpu'))
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

desc = ["A chair with three legs."]
vocab = Vocabulary.create_word_vocab([x.split(" ") for x in desc], 100)
tokenizer = Tokenizer(vocab)
lang_model = ToyLanguageEncoder(tokenizer)

all_ref = []
all_recons = []
all_lang_recons = []

for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        lang_code = lang_model.encode(desc)
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

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())
np.save(os.path.join(save_dir, 'lang_out.npy'), all_lang_recons.numpy())


