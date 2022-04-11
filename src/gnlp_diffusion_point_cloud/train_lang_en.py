import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.misc import *
from utils.data import *
from utils.transform import *
from utils.dataset import *
from models.language_encoder import *

from pathlib import Path
from utilities.paths import DATA_FOLDER


# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--size', type=str, default="small", choices=["small", "base", "large", "extra_large", "largest"])
parser.add_argument('--model', type=str, default="T5", choices=["T5"])
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--token_length', type=int, default=128)

# Datasets and loaders
parser.add_argument('--lang_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_text_data.hdf5")
parser.add_argument('--emb_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_embeddings_data.hdf5")
parser.add_argument('--categories', type=str_list, default=['table'])
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=32)

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
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

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
    model = LanguageEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = LanguageEncoder(args).to(args.device)
logger.info(repr(model))

# Datasets and loaders
transform = None
logger.info('Loading datasets...')
train_dset = NLShapeNetCoreEmbeddings(
    desc_path=args.lang_dataset_path,
    embedding_path=args.emb_dataset_path,
    cates=args.categories,
    split='train',
    tokenizer=model.tokenizer,
    token_length = args.token_length,
    transform=transform,
)
val_dset = NLShapeNetCoreEmbeddings(
    desc_path=args.lang_dataset_path,
    embedding_path=args.emb_dataset_path,
    cates=args.categories,
    split='val',
    tokenizer=model.tokenizer,
    token_length = args.token_length,
    transform=transform,
)
train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
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
    batch_tokens, batch_encodings = next(train_iter)
    x = batch_tokens.to(args.device)
    y = batch_encodings.to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x, y)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()

def validate_loss(it):
    return
    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()

    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    return cd

def validate_inspect(it):
    return;
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break   # Inspect only 5 batch

    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
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
            cd_loss = 0
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
