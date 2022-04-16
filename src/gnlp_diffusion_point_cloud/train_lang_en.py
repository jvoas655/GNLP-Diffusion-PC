import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from IPython.display import Image
import cv2
import matplotlib.pyplot as plt

from utils.misc import *
from utils.data import *
from utils.transform import *
from utils.dataset import *
from models.language_encoder import *
from diffusion_point_cloud.models.vae_flow import *
from diffusion_point_cloud.models.vae_gaussian import *
from diffusion_point_cloud.models.autoencoder import *
from evaluation.evaluation_metrics import EMD_CD
from diffusion_point_cloud.models.autoencoder import *

from pathlib import Path
from utilities.paths import DATA_FOLDER, PRETRAINED_FOLDER
from utilities.visualizations import save_pc_drawing


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--size', type=str, default="small", choices=["small", "base", "large", "extra_large", "largest"])
parser.add_argument('--backbone', type=str, default="T5", choices=["T5", "SIMPLE"])
parser.add_argument('--encoder', type=str, default="CNN2FF", choices=["CNN2FF", "SIMPLE"])
parser.add_argument('--loss', type=str, default="DiffusionMSE", choices=[
    "MSE", "ContrastiveCos", "ContrastiveLoss", "DiffusionMSE", "DiffusionGenMSE"
])
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--token_length', type=int, default=128)
parser.add_argument('--contrastive_size', type=int, default=32)
parser.add_argument('--dataset_size', '-ds', type=int, default=-1,
                    help='-1 means all dataset examples are used, anything higher will slice the dataset.')
parser.add_argument('--use_train_as_validation', action='store_true', dest='use_train_as_validation',
                    help='Use the training dataloader as the validation dataset, useful for debugging.')
parser.add_argument('--visualize', '-v', action='store_true', dest='visualize')
parser.add_argument('--skip_checkpoint', action='store_true', dest='skip_checkpoint')

# Datasets and loaders
parser.add_argument('--lang_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_text_data.hdf5")
parser.add_argument('--emb_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_embeddings_data.hdf5")
parser.add_argument('--pc_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_pc_data.hdf5")
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--gen_ckpt', type=str, default= PRETRAINED_FOLDER / "AE_all.pt")
parser.add_argument('--gen_model', type=str, default='AE', choices=['flow', 'gaussian', 'AE'])
parser.add_argument('--gen_flexibility', type=float, default=0.0)
parser.add_argument('--gen_sample_num_points', type=int, default=2048)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--backbone_freeze', type=eval, default=False, choices=[True, False])
parser.add_argument('--scale_mode', type=str, default='shape_unit')

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--val_type', type=str, default='CDEmbeddings', choices=['CDEmbeddings', 'CDTruePC'])
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

visualize = args.visualize
skip_checkpoint = args.skip_checkpoint

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='LE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = LanguageEncoder(**vars(ckpt['args'])).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = LanguageEncoder(**vars(args)).to(args.device)

logger.info(repr(model))

logger.info('Building validation model...')

# Helpful for people on devices without gpus for debugging
if args.device == 'cpu':
    gen_ckpt = torch.load(args.gen_ckpt, map_location=torch.device('cpu'))
else:
    gen_ckpt = torch.load(args.gen_ckpt)

if args.gen_model == 'AE':
  gen_model = AutoEncoder(gen_ckpt['args']).to(args.device)
  gen_model.sample = gen_model.decode
elif args.gen_model == 'gaussian':
    gen_model = GaussianVAE(gen_ckpt['args']).to(args.device)
elif args.gen_model == 'flow':
    gen_model = FlowVAE(gen_ckpt['args']).to(args.device)

gen_model.load_state_dict(gen_ckpt['state_dict'])
logger.info(repr(gen_model))
#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#rint(params)
# Datasets and loaders
transform = None
logger.info('Loading datasets...')
train_dset = NLShapeNetCoreEmbeddings(
    desc_path=args.lang_dataset_path,
    embedding_path=args.emb_dataset_path,
    pc_path=args.pc_dataset_path,
    cates=args.categories,
    split='train',
    tokenizer=model.backbone.tokenizer,
    token_length=args.token_length,
    scale_mode=args.scale_mode,
    transform=transform,
    size=args.dataset_size,
)
val_dset = NLShapeNetCoreEmbeddings(
    desc_path=args.lang_dataset_path,
    embedding_path=args.emb_dataset_path,
    pc_path=args.pc_dataset_path,
    cates=args.categories,
    split='val',
    tokenizer=model.backbone.tokenizer,
    token_length=args.token_length,
    scale_mode=args.scale_mode,
    transform=transform,
    size=args.dataset_size,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))

if args.use_train_as_validation:
    val_loader = DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
)
else:
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
steps = int((args.sched_end_epoch - args.sched_start_epoch) / 10)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[args.sched_start_epoch for offset in range(0, args.sched_end_epoch - args.sched_start_epoch, steps)],
    gamma=(args.end_lr / args.lr) ** (1/10)
)


# Train, validate
def train(it):
    # Load data
    batch_tokens, batch_encodings, batch_pcs = next(train_iter)
    if (args.loss == "ContrastiveCos"):
        shuffle_size = min(int(batch_tokens.shape[0] / 2), args.contrastive_size)
        rand_start = random.randint(0, batch_tokens.shape[0] - shuffle_size)
        targets = np.ones(batch_tokens.shape[0])
        inds = np.arange(shuffle_size)
        rand_inds = np.random.shuffle(inds)
        batch_tokens[rand_start:rand_start + shuffle_size] = batch_tokens[rand_start:rand_start + shuffle_size, :][rand_inds, :]
        targets[rand_start:rand_start + shuffle_size] = np.where(inds != rand_inds, -1, 1)
        targets = torch.Tensor(targets).to(args.device)
    x = batch_tokens.to(args.device)
    y = batch_encodings.to(args.device)
    batch_pcs = batch_pcs.to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    if args.loss == "DiffusionMSE":
        loss = model.get_loss(x, y, pcs=batch_pcs)
    elif args.loss == "ContrastiveCos":
        loss = model.get_loss(x, y, targets=targets)
    elif args.loss == "DiffusionGenMSE":
        loss = model.get_loss(x, y, pcs=batch_pcs, gen_model=gen_model)
    else:
        loss = model.get_loss(x, y)

    # Backward and optimize
    loss.backward()
    # orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f' % (it, loss.item()))
    # writer.add_scalar('train/loss', loss, it)
    # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    # writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):
    all_refscons = []
    all_embcons = []
    all_modcons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref_tokens, ref_embeddings, ref_pc = batch
        ref_tokens=ref_tokens.to(args.device)
        ref_embeddings=ref_embeddings.to(args.device)
        ref_pc=ref_pc.to(args.device)
        with torch.no_grad():
            # model.eval()
            code = model.encode(ref_tokens)
            # gen_model.eval()
            # seed = torch.seed()
            #if (args.val_type == "CDEmbeddings"):
            all_embcons.append(gen_model.sample(ref_embeddings, args.gen_sample_num_points, flexibility=args.gen_flexibility).detach())
            # torch.manual_seed(seed)
            #elif (args.val_type == "CDTruePC"):
            all_modcons.append(gen_model.sample(code, args.gen_sample_num_points, flexibility=args.gen_flexibility).detach())
            all_refscons.append(ref_pc)

    all_refscons = torch.cat(all_refscons, dim=0)
    all_modcons = torch.cat(all_modcons, dim=0)
    all_embcons = torch.cat(all_embcons, dim=0)
    mod_metrics = EMD_CD(all_modcons, all_refscons, batch_size=args.val_batch_size)
    mod_cd, mod_emd = mod_metrics['MMD-CD'].item(), mod_metrics['MMD-EMD'].item()
    emb_metrics = EMD_CD(all_embcons, all_refscons, batch_size=args.val_batch_size)
    emb_cd, emb_emd = emb_metrics['MMD-CD'].item(), emb_metrics['MMD-EMD'].item()

    logger.info('[Val] Iter %04d | MOD_CD %.6f | EMB_CD %.6f  ' % (it, mod_cd, emb_cd))
    # writer.add_scalar('val/mod_cd', mod_cd, it)
    # writer.add_scalar('val/emb_cd', emb_cd, it)
    writer.flush()

    if visualize:
        save_pc_drawing(all_modcons.detach().cpu().numpy(), "Models")

    return mod_cd

def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        ref_tokens, ref_embeddings, ref_pc = batch
        ref_tokens = ref_tokens.to(args.device)
        ref_embeddings = ref_embeddings.to(args.device)
        ref_pc = ref_pc.to(args.device)
        model.eval()
        code = model.encode(ref_tokens)
        recons = gen_model.sample(code, args.gen_sample_num_points, flexibility=args.gen_flexibility).detach().cpu()

        sum_n += ref_tokens.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    # writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    writer.flush()

# Main loop
logger.info('Start training...')
try:
    it = 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }

            if not skip_checkpoint:
                ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
