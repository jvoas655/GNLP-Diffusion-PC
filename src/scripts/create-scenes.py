from argparse import ArgumentParser
from pathlib import Path
import gdown
import h5py
import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
import random

#from utilities.paths import DATA_FOLDER
DATA_FOLDER = Path("C:\\Users\\Jorda\\Documents\\School\\Spring2022\\CS395T\\Project\\Dup\\GNLP-Diffusion-PC\\src\\data")

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

def display(title, data):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection="3d")
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    ax.set_box_aspect((np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])))
    #fig.tight_layout()
    fig.suptitle(title)
    plt.show()
    plt.close(fig)

def gen_scene(args, text, pc):
    # pick two objects randomly
    cate_1 = random.choice(["chair", "table"])
    obj_1_ind = random.randint(0, text[cate_to_synsetid[cate_1]].shape[0])
    sent_1 = text[cate_to_synsetid[cate_1]][obj_1_ind]
    data_1 = pc[cate_to_synsetid[cate_1]][obj_1_ind, :, :]
    # Select object modifiers
    roll = random.random()
    tip_overs = ["left side", "back side", "front side", "right side", "feet"]
    if (roll < 0.02):
        base_strs = ["{obj} laying on it {side}"]
        tip = random.choice(tip_overs)
        sent_1 = random.choice(base_strs).format(obj=sent_1, side=tip)
        if (tip == "left side"):
            store = data_1[:, 1].copy()
            data_1[:, 1] = data_1[:, 0]
            data_1[:, 0] = store
            data_1[:, 0] = np.max(data_1[:, 0]) - data_1[:, 0]
        elif (tip == "right side"):
            store = data_1[:, 1].copy()
            data_1[:, 1] = data_1[:, 0]
            data_1[:, 0] = store
        elif (tip == "front side"):
            store = data_1[:, 1].copy()
            data_1[:, 1] = data_1[:, 2]
            data_1[:, 2] = store
        elif (tip == "back side"):
            store = data_1[:, 1].copy()
            data_1[:, 1] = data_1[:, 2]
            data_1[:, 2] = store
            data_1[:, 2] = np.max(data_1[:, 2]) - data_1[:, 2]
    obj_1_bounds = np.zeros((3, 2)) # dim, min/max
    for dim in range(3):
        obj_1_bounds[dim, :] = [data_1[:, dim].min(), data_1[:, dim].max()]
        data_1[:, dim] -= obj_1_bounds[dim, 0]


    cate_2 = random.choice(["chair", "table"])
    obj_2_ind = random.randint(0, text[cate_to_synsetid[cate_2]].shape[0])
    sent_2 = text[cate_to_synsetid[cate_2]][obj_2_ind]
    data_2 = pc[cate_to_synsetid[cate_2]][obj_2_ind, :, :]
    roll = random.random()
    tip_overs = ["left side", "back side", "front side", "right side", "feet"]
    if (roll < 0.02):
        base_strs = ["{obj} laying on it {side}"]
        tip = random.choice(tip_overs)
        sent_2 = random.choice(base_strs).format(obj=sent_2, side=tip)
        if (tip == "left side"):
            store = data_2[:, 1]
            data_2[:, 1] = data_2[:, 0]
            data_2[:, 0] = store
            data_2[:, 0] = np.max(data_2[:, 0]) - data_2[:, 0]
        elif (tip == "right side"):
            store = data_2[:, 1]
            data_2[:, 1] = data_2[:, 0]
            data_2[:, 0] = store
        elif (tip == "front side"):
            store = data_2[:, 1]
            data_2[:, 1] = data_2[:, 2]
            data_2[:, 2] = store
        elif (tip == "back side"):
            store = data_2[:, 1]
            data_2[:, 1] = data_2[:, 2]
            data_2[:, 2] = store
            data_2[:, 2] = np.max(data_2[:, 2]) - data_2[:, 2]
    obj_2_bounds = np.zeros((3, 2)) # dim, min/max
    for dim in range(3):
        obj_2_bounds[dim, :] = [data_2[:, dim].min(), data_2[:, dim].max()]
        data_2[:, dim] -= obj_2_bounds[dim, 0]
    # floor
    if (args.floor_spacing):
        floor_point_locs = np.meshgrid(np.linspace(0, 10, args.floor_spacing), np.zeros(1), np.linspace(0, 10, args.floor_spacing))
        floor = np.zeros((args.floor_spacing * args.floor_spacing, 3))
        for dim in range(3):
            floor[:, dim] = floor_point_locs[dim].flatten()


    # Select combining method
    roll = random.random()
    segments = ["center", "right", "lower right", "upper right", "bottom", "top", "left", "upper left", "lower left"]
    offsets = [(3, 3), (0, 3), (0, 6), (0, 0), (3, 6), (3, 0), (6, 3), (6, 0), (6, 6)]
    referential = ["slightly to the right of", "to the right of", \
        "slightly below", "below", "slightly above", \
        "above", "slightly to the left of", "to the left of"]
    referential_offsets = [(-0.5, 0), (-2, 0), (0, 0.5), (0, 2), (0, -0.5), (0, -2), (0.5, 0), (2, 0)]
    if (roll < 1):
        pos = 0
        ref = random.randint(0, len(referential)-1)
        base_strs = ["A {obj_2} is {dist} a {obj_1}"]
        defn_str = random.choice(base_strs).format(obj_1 = sent_1, obj_2 = sent_2, dist = referential[ref])
        x_off, z_off = referential_offsets[ref]
        rx = np.clip(np.random.normal(), a_min=-0.5, a_max = 0.5)
        rz = np.clip(np.random.normal(), a_min=-0.5, a_max = 0.5)
        data_1[:, 0] += 4 + rx
        data_1[:, 2] += 4 + rz

        data_2[:, 0] += 4 + rx
        data_2[:, 2] += 4 + rz
        if (x_off < 0):
            data_2[:, 0] -= np.max(data_2[:, 0]) - np.min(data_1[:, 0])
        elif (x_off > 0):
            data_2[:, 0] += np.max(data_1[:, 0]) - np.min(data_2[:, 0])
        elif (z_off < 0):
            data_2[:, 2] -= np.max(data_2[:, 2]) - np.min(data_1[:, 2])
        else:
            data_2[:, 2] += np.max(data_1[:, 2]) - np.min(data_2[:, 2])
        data_2[:, 0] += x_off
        data_2[:, 2] += z_off
        fused_data = np.concatenate((data_1, data_2))
        if (args.floor_spacing):
            fused_data = np.concatenate((floor, fused_data))
        fused_data[:, 0] -= 5
        fused_data[:, 2] -= 5
    else:
        # Scene referential
        pos1 = random.randint(0, len(segments)-1)
        pos2 = random.randint(0, len(segments)-1)
        while (pos2 == pos1):
            pos2 = random.randint(0, len(segments))
        base_strs = ["A {obj_1} is in the {obj_1_pos} of the room with a {obj_2} in the {obj_2_pos}",\
                    "A {obj_1} is placed near the {obj_1_pos} of the room with a {obj_2} to the {obj_2_pos}"]
        defn_str = random.choice(base_strs).format(obj_1 = sent_1, obj_1_pos = segments[pos1], obj_2 = sent_2, obj_2_pos = segments[pos2])
        data_1[:, 0] += offsets[pos1][0]
        data_1[:, 2] += offsets[pos1][1]
        data_2[:, 0] += offsets[pos2][0]
        data_2[:, 2] += offsets[pos2][1]
        fused_data = np.concatenate((data_1, data_2))
        fused_bounds = np.zeros((3, 2)) # dim, min/max
        for dim in range(3):
            fused_bounds[dim, :] = [fused_data[:, dim].min(), fused_data[:, dim].max()]
        fused_data[:, 0] += np.clip(np.random.normal(1), a_min=0, a_max=3 - max(obj_1_bounds[0, 1], obj_2_bounds[0, 1]))
        fused_data[:, 2] += np.clip(np.random.normal(1), a_min=0, a_max=3 - max(obj_1_bounds[2, 1], obj_2_bounds[2, 1]))
        if (args.floor_spacing):
            fused_data = np.concatenate((floor, fused_data))
        fused_data[:, 0] -= 5
        fused_data[:, 2] -= 5
    if (args.inspect):
        display(defn_str, fused_data)
    return defn_str, fused_data

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('--input_pc', type=str,
                           help='',
                           default=DATA_FOLDER / "aligned_pc_data.hdf5")
    argparser.add_argument('--input_text', type=str,
                           help='',
                           default=DATA_FOLDER / "aligned_text_data.hdf5")
    argparser.add_argument('--output_pc', type=str,
                           help='',
                           default=DATA_FOLDER / "aligned_scene_pc_data.hdf5")
    argparser.add_argument('--output_text', type=str,
                           help='',
                           default=DATA_FOLDER / "aligned_scene_text_data.hdf5")
    argparser.add_argument('--num_train', type=int,
                           help='',
                           default=7000)
    argparser.add_argument('--num_val', type=int,
                           help='',
                           default=1500)
    argparser.add_argument('--num_test', type=int,
                           help='',
                           default=1500)
    argparser.add_argument('--min_words', type=int,
                           help='',
                           default=4)
    argparser.add_argument('--floor_spacing', type=int,
                           help='',
                           default=16)
    argparser.add_argument('--inspect',
                           help='',
                           default=True,
                           action = "store_true")

    args = argparser.parse_args()

    stop_words = set(nltk.corpus.stopwords.words("english"))

    pc_data = h5py.File(args.input_pc)
    text_data = h5py.File(args.input_text)
    all_pcs = {}
    all_text = {}
    all_text_processed = {}
    all_pcs_processed = {}

    for cate in pc_data:
        print(synsetid_to_cate[cate])
        all_pcs[cate] = []
        all_text[cate] = []
        for split in pc_data[cate]:
            all_pcs[cate].append(pc_data[cate][split][()])
            all_text[cate].append(text_data[cate][split][()])
        all_pcs[cate] = np.concatenate(all_pcs[cate])
        all_text[cate] = np.concatenate(all_text[cate])
        for text in range(all_text[cate].shape[0]):
            all_text[cate][text] = all_text[cate][text].decode("utf-8").lower()
        print(all_pcs[cate].shape)
        print(all_text[cate].shape)
        all_text_processed[cate] = all_text[cate].copy()
        all_pcs_processed[cate] = all_pcs[cate].copy()
    num_valid = 0
    for i in range(all_text[cate_to_synsetid["chair"]].shape[0]):
        #print(all_text[cate_to_synsetid["table"]][i])
        words = [word for sent in nltk.sent_tokenize(all_text[cate_to_synsetid["chair"]][i]) for word in nltk.word_tokenize(sent)]
        #words = nltk.pos_tag(words)
        words = [w for w in words if not w in stop_words and w.isalpha()]
        if (len(words) < args.min_words):
            continue
        new_sent = " ".join(words)
        if ("chair" in new_sent):
            split = new_sent.split("chair")
            if (split[-1] != ""):
                new_sent = "chair with".join(split)
        if ("color" in new_sent):
            split = new_sent.split("color")
            if (split[-1] != ""):
                new_sent = "color and ".join(split)
        new_sent = new_sent.replace("  ", " ").strip()
        all_text_processed[cate_to_synsetid["chair"]][num_valid] = new_sent
        all_pcs_processed[cate_to_synsetid["chair"]][num_valid] = all_pcs[cate_to_synsetid["chair"]][i]
        num_valid += 1
    all_pcs_processed[cate_to_synsetid["chair"]] = all_pcs_processed[cate_to_synsetid["chair"]][:num_valid, :, :]
    all_text_processed[cate_to_synsetid["chair"]] = all_text_processed[cate_to_synsetid["chair"]][:num_valid]
    num_valid = 0
    for i in range(all_text[cate_to_synsetid["table"]].shape[0]):
        #print(all_text[cate_to_synsetid["table"]][i])
        words = [word for sent in nltk.sent_tokenize(all_text[cate_to_synsetid["table"]][i]) for word in nltk.word_tokenize(sent)]
        #words = nltk.pos_tag(words)
        words = [w for w in words if not w in stop_words and w.isalpha()]
        if (len(words) < args.min_words):
            continue
        new_sent = " ".join(words)
        if ("table" in new_sent):
            split = new_sent.split("table")
            if (split[-1] != ""):
                new_sent = "table with".join(split)
        if ("desk" in new_sent):
            split = new_sent.split("desk")
            if (split[-1] != ""):
                new_sent = "desk with".join(split)
        if ("color" in new_sent):
            split = new_sent.split("color")
            if (split[-1] != ""):
                new_sent = "color and ".join(split)
        new_sent = new_sent.replace("  ", " ").strip()
        all_text_processed[cate_to_synsetid["table"]][num_valid] = new_sent
        all_pcs_processed[cate_to_synsetid["table"]][num_valid] = all_pcs[cate_to_synsetid["table"]][i]
        num_valid += 1
    all_pcs_processed[cate_to_synsetid["table"]] = all_pcs_processed[cate_to_synsetid["table"]][:num_valid, :, :]
    all_text_processed[cate_to_synsetid["table"]] = all_text_processed[cate_to_synsetid["table"]][:num_valid]
    print(all_text_processed[cate_to_synsetid["table"]].shape)
    print(all_text_processed[cate_to_synsetid["chair"]].shape)
    train_text = []
    train_pc = []
    for n in range(args.num_train):
        text, pc = gen_scene(args, all_text_processed, all_pcs_processed)
        train_text.append(text.encode("utf-8"))
        train_pc.append(pc)
    train_text = np.array(train_text)
    print(train_text.shape)
    train_pc = np.array(train_pc)
    print(train_pc.shape)
    val_text = []
    val_pc = []
    for n in range(args.num_val):
        text, pc = gen_scene(args, all_text_processed, all_pcs_processed)
        val_text.append(text.encode("utf-8"))
        val_pc.append(pc)
    val_text = np.array(val_text)
    print(val_text.shape)
    val_pc = np.array(val_pc)
    print(train_pc.shape)
    test_text = []
    test_pc = []
    for n in range(args.num_test):
        text, pc = gen_scene(args, all_text_processed, all_pcs_processed)
        test_text.append(text.encode("utf-8"))
        test_pc.append(pc)
    test_text = np.array(test_text)
    print(test_text.shape)
    test_pc = np.array(test_pc)
    print(test_pc.shape)



    if (not args.inspect):
        out_pc_data = h5py.File(args.output_pc, "w")
        out_text_data = h5py.File(args.output_text, "w")
        for cate in pc_data:
            out_pc_data.create_group(cate)
            out_text_data.create_group(cate)
            out_pc_data[cate].create_dataset("train", data = train_pc)
            out_text_data[cate].create_dataset("train", data = train_text)
            out_pc_data[cate].create_dataset("val", data = val_pc)
            out_text_data[cate].create_dataset("val", data = val_text)
            out_pc_data[cate].create_dataset("test", data = test_pc)
            out_text_data[cate].create_dataset("test", data = test_text)
