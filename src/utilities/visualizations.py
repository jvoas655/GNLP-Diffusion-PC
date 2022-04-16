import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import math


def __draw_pc__(data, title, outfile=None):
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

    return plt


def draw_pc(data, title):
    __draw_pc__(data, title)
    plt.show()


def save_pc_drawing(data, title, outfile='./vis.png'):
    __draw_pc__(data, title)
    plt.savefig(outfile)
