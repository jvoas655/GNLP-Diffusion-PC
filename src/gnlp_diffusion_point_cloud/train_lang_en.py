import os
import sys

import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from IPython.display import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.misc import *
from utils.data import *
from utils.transform import *
from utils.dataset import *
from models.language_encoder import *
from models.full_encoder_decoder import *
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
parser.add_argument('--loss_weights', nargs='+', type=float, default=[0.5, 1.0, 0.5],
                    help='When multiple losses are chosen, you can specify how much impact each loss has by passing in'
                         ' a weight value for each loss in order of the losses specified by --losses')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--token_length', type=int, default=64)
parser.add_argument('--dataset_size', '-ds', type=int, default=-1,
                    help='-1 means all dataset examples are used, anything higher will slice the dataset.')
parser.add_argument('--use_train_as_validation', action='store_true', dest='use_train_as_validation', default=False,
                    help='Use the training dataloader as the validation dataset, useful for debugging.')
parser.add_argument('--visualize', '-v', action='store_true', dest='visualize')
parser.add_argument('--skip_checkpoint', action='store_true', dest='skip_checkpoint')

# Datasets and loaders
parser.add_argument('--lang_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_text_data.hdf5")
parser.add_argument('--pc_dataset_path', type=str, default=Path(DATA_FOLDER) / "aligned_pc_data.hdf5")
parser.add_argument('--categories', type=str_list, default=["chair", "table"], help="all, chair, table, or scenes (alone)")
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--gen_ckpt', type=str, default= None)
parser.add_argument('--gen_model', type=str, default='flow', choices=['flow', 'gaussian', 'AE'])
parser.add_argument('--gen_flexibility', type=float, default=0.0)
parser.add_argument('--gen_sample_num_points', type=int, default=2048) # 4352
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--backbone_freeze', type=eval, default=False, choices=[True, False])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--multi_description_per_object', '-mdpo', action='store_true', dest='multi_description_per_object',
                    help='Used for when your aligned_text file has ||| delimiting multiple descriptions for the same '
                         'PC', default=False)
parser.add_argument('--multi_description_sample_method', '-mdsm', choices=['sample', 'combined'],
                    type=str, default='sample')
parser.add_argument('--log_data', action='store_true',
                    help='', default=True)

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
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--val_type', type=str, default='CDEmbeddings', choices=['CDEmbeddings', 'CDTruePC'])
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=6)
parser.add_argument('--num_inspect_pointclouds', type=int, default=3)
args = parser.parse_args()
seed_all(args.seed)

visualize = args.visualize
skip_checkpoint = args.skip_checkpoint

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='LEGEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)
if (args.log_data):
    os.mkdir(log_dir + "\\figures")
# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = FullEncoderDecoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = FullEncoderDecoder(args).to(args.device)

logger.info(repr(model))


#model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#params = sum([np.prod(p.size()) for p in model_parameters])
#rint(params)
# Datasets and loaders
transform = None
logger.info('Loading datasets...')
train_dset = NLShapeNetCore(
    desc_path=args.lang_dataset_path,
    pc_path=args.pc_dataset_path,
    cates=args.categories,
    split='train',
    tokenizer=model.backbone.tokenizer,
    token_length=args.token_length,
    scale_mode=args.scale_mode,
    transform=transform,
    size=args.dataset_size,
    multi_description_per_object=args.multi_description_per_object,
    multi_description_sample_method=args.multi_description_sample_method
)
val_dset = NLShapeNetCore(
    desc_path=args.lang_dataset_path,
    pc_path=args.pc_dataset_path,
    cates=args.categories,
    split='val',
    tokenizer=model.backbone.tokenizer,
    token_length=args.token_length,
    scale_mode=args.scale_mode,
    transform=transform,
    size=args.dataset_size,
    multi_description_per_object=args.multi_description_per_object,
    multi_description_sample_method=args.multi_description_sample_method
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

train_dset[0]
def save_render(path, it, ind, title, data):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_box_aspect((np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])))
    #fig.tight_layout()
    ax.set_title(title.format(it=it, ind=ind))
    plt.savefig(path+"\\"+title.format(it=it, ind=ind))
    plt.close(fig)
# Train, validate
loss_store = []
epi_loss_store = []
epi_iters = []

def train(it):
    global loss_store
    # Load data
    batch_tokens, batch_pcs = next(train_iter)
    x = batch_tokens.to(args.device)
    batch_pcs = batch_pcs.to(args.device, dtype=torch.float)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x, batch_pcs, args.loss_weights)
    # Backward and optimize
    loss.backward()
    # orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f' % (it, loss.item()))
    if (args.log_data):
        loss_store.append(loss.item())
    # writer.add_scalar('train/loss', loss, it)
    # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    # writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()
    if args.log_data and (it % args.val_freq == 0 or it == args.max_iters):
        global epi_loss_store
        global epi_iters
        epi_iters.append(it)
        epi_loss_store.append(np.mean(loss_store))
        loss_store = []
        plt.plot(epi_iters, epi_loss_store)
        plt.title("Train Loss")
        plt.xlabel("Train Iteration")
        plt.ylabel("Loss")
        plt.savefig(log_dir + "\\figures"+"\\TRAIN_LOSS.png")
        plt.close()
        all_refscons = []
        all_pccons = []
        all_textcons = []
        all_textcons = model.decode(model.encode_text(x), args.gen_sample_num_points).detach()
        all_pccons = model.decode(model.encode_pc(batch_pcs), args.gen_sample_num_points).detach()
        all_refscons = batch_pcs.detach()
        for i in range(args.num_inspect_pointclouds):
            save_render(log_dir + "\\figures", it, i, "TRAIN_REF{it}_{ind}", all_refscons[i, :, :].cpu().detach().numpy())
            save_render(log_dir + "\\figures", it, i, "TRAIN_PC{it}_{ind}", all_pccons[i, :, :].cpu().detach().numpy())
            save_render(log_dir + "\\figures", it, i, "TRAIN_TEXT{it}_{ind}", all_textcons[i, :, :].cpu().detach().numpy())
val_iters = []
text_cds = []
pc_cds = []

def validate_loss(it):
    global val_iters
    global text_cds
    global pc_cds
    all_refscons = []
    all_pccons = []
    all_textcons = []
    codes_pc = []
    codes_text = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref_tokens, ref_pc = batch
        ref_tokens=ref_tokens.to(args.device)
        ref_pc=ref_pc.to(args.device, dtype=torch.float)
        with torch.no_grad():
            # model.eval()
            code_text = model.encode_text(ref_tokens)
            code_pc = model.encode_pc(ref_pc)
            if (args.log_data):
                codes_pc.append(code_pc)
                codes_text.append(code_text)
            all_textcons.append(model.decode(code_text, args.gen_sample_num_points).detach())
            all_pccons.append(model.decode(code_pc, args.gen_sample_num_points).detach())
            all_refscons.append(ref_pc)

    all_refscons = torch.cat(all_refscons, dim=0)
    all_pccons = torch.cat(all_pccons, dim=0)
    all_textcons = torch.cat(all_textcons, dim=0)
    if (args.log_data):
        codes_pc = torch.cat(codes_pc, dim=0)
        codes_text = torch.cat(codes_text, dim=0)
    if (args.log_data):
        pca = PCA().fit(codes_pc.cpu().detach().numpy())
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("PC PCA{it}".format(it=it))
        plt.savefig(log_dir + "\\figures"+"\\"+"PC_PCA{it}.png".format(it=it))
        plt.close()
        pca = PCA().fit(codes_text.cpu().detach().numpy())
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title("TEXT PCA{it}".format(it=it))
        plt.savefig(log_dir + "\\figures"+"\\"+"TEXT_PCA{it}.png".format(it=it))
        plt.close()
    text_metrics = EMD_CD(all_textcons, all_refscons, batch_size=args.val_batch_size)
    for i in range(args.num_inspect_pointclouds):
        if (it == args.val_freq):
            save_render(log_dir + "\\figures", it, i, "REF{it}_{ind}", all_refscons[i, :, :].cpu().detach().numpy())
        save_render(log_dir + "\\figures", it, i, "PC{it}_{ind}", all_pccons[i, :, :].cpu().detach().numpy())
        save_render(log_dir + "\\figures", it, i, "TEXT{it}_{ind}", all_textcons[i, :, :].cpu().detach().numpy())
    text_cd, text_emd = text_metrics['MMD-CD'].item(), text_metrics['MMD-EMD'].item()
    text_cds.append(text_cd)
    pc_metrics = EMD_CD(all_pccons, all_refscons, batch_size=args.val_batch_size)
    pc_cd, pc_emd = pc_metrics['MMD-CD'].item(), pc_metrics['MMD-EMD'].item()
    if (args.log_data):
        pc_cds.append(pc_cd)
        val_iters.append(it)
        plt.plot(val_iters, text_cds)
        plt.title("Text Val CD")
        plt.xlabel("Val Iteration")
        plt.ylabel("CD")
        plt.savefig(log_dir + "\\figures"+"\\TEXT_VAL_CD.png")
        plt.close()
        plt.plot(val_iters, pc_cds)
        plt.title("PC Val CD")
        plt.xlabel("Val Iteration")
        plt.ylabel("CD")
        plt.savefig(log_dir + "\\figures"+"\\PC_VAL_CD.png")
        plt.close()
    logger.info('[Val] Iter %04d | TEXT_CD %.6f | PC_CD %.6f  ' % (it, text_cd, pc_cd))
    # writer.add_scalar('val/mod_cd', mod_cd, it)
    # writer.add_scalar('val/emb_cd', emb_cd, it)
    writer.flush()

    if visualize:
        save_pc_drawing(all_modcons.detach().cpu().numpy(), "Models")

    return text_cd

def validate_inspect(it):
    return
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        ref_tokens, ref_pc = batch
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
