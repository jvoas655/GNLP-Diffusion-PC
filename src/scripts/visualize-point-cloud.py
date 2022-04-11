import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import math
import argparse
import h5py
from pathlib import Path
import torch
from diffusionpointcloud.models.autoencoder import *
from diffusionpointcloud.models.vae_gaussian import *
from diffusionpointcloud.models.vae_flow import *
from utilities.paths import DATA_FOLDER


def draw_pc(data, title, outfile=None):
    num_sub_plots = 1
    if (len(data.shape) == 3):
        num_sub_plots = data.shape[0]
    dim_size = math.ceil(math.sqrt(num_sub_plots))
    fig = plt.figure()
    if (num_sub_plots == 1):
        data = data.reshape(1, -1, 3)
    for ax_num in range(num_sub_plots):
        ax = plt.subplot(dim_size, dim_size, ax_num + 1, projection="3d")
        ax.scatter(data[ax_num, :, 0], data[ax_num, :, 1], data[ax_num, :, 2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    fig.tight_layout()
    fig.suptitle(title)
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str, default='', help="Input file containing [N, K, 3] data. Can be a .npy or .hdf5 file. If left blank will geneate a random [1, 2048, 3] point cloud")
parser.add_argument("--mode", type=str, default="raw", help="['raw', 'AE', 'AE_GEN'] Will specify whether to show the model as is or after processing through a model. Muitiple can be given comma separated")
parser.add_argument("--ae_path", type=str, default='')
parser.add_argument("--gen_path", type=str, default='')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--gen_model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--normalize', type=str, default='shape_bbox', choices=[None, 'shape_unit', 'shape_bbox'])
parser.add_argument("--start_ind", type=int, default=-1)
parser.add_argument("--stop_ind", type=int, default=-1)
args = parser.parse_args()
args.mode = args.mode.split(",")

if (args.start_ind != -1 and args.stop_ind != -1):
    raw_data = np.load(args.input_data)[args.start_ind:args.stop_ind, :, :]
elif (args.start_ind != -1):
    raw_data = np.load(args.input_data)[args.start_ind:, :, :]
else:
    raw_data = np.load(args.input_data)[:args.stop_ind, :, :]
if (args.normalize == "shape_bbox"):
    for ind in range(raw_data.shape[0]):
        pc_max = raw_data[ind, :, :].max(axis=0, keepdims=True) # (1, 3)
        pc_min = raw_data[ind, :, :].min(axis=0, keepdims=True) # (1, 3)
        shift = ((pc_min + pc_max) / 2).reshape(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        raw_data[ind, :, :] = (raw_data[ind, :, :] - shift) / scale
data_set = []
if ("raw" in args.mode):
    data_set.append(raw_data)
if (("AE" in args.mode or "AE_GEN" in args.mode) and args.ae_path != ""):
    ckpt = torch.load(args.ae_path)
    ae_model = AutoEncoder(ckpt['args']).to(args.device)
    ae_model.load_state_dict(ckpt['state_dict'])
    ae_model.eval()
    encoded_data = ae_model.encode(torch.tensor(raw_data).cuda())
    decoded_data = ae_model.decode(encoded_data, raw_data.shape[1])
    if ("AE" in args.mode):
        data_set.append(decoded_data.clone().detach().cpu().numpy())

if ("AE_GEN" in args.mode and args.gen_path != ""):
    ckpt = torch.load(args.gen_path)
    if args.gen_model == 'gaussian':
        gen_model = GaussianVAE(ckpt['args']).to(args.device)
    elif args.gen_model == 'flow':
        gen_model = FlowVAE(ckpt['args']).to(args.device)
    gen_model.load_state_dict(ckpt['state_dict'])
    gen_model.eval()
    dif_data = gen_model.sample(encoded_data, raw_data.shape[1], flexibility=ckpt['args'].flexibility)
    data_set.append(dif_data.clone().detach().cpu().numpy())
data = np.array(data_set).transpose(1, 0, 2, 3).reshape(-1, raw_data.shape[1], 3)



draw_pc(data, "a")
