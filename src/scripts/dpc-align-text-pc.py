from argparse import ArgumentParser
from pathlib import Path
import gdown
import h5py
import pandas as pd
import os
import json
import numpy as np

from utilities.paths import DIFFUSION_MODEL_DATA_FOLDER

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

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('-o', '--output', type=str,
                           help='Shapenet File location, default is (Diffusion Models Directory under data/)',
                           default=str(DIFFUSION_MODEL_DATA_FOLDER))

    args = argparser.parse_args()

    text_data = pd.read_csv(Path(args.output) / "text2shape.captions.csv")[["modelId", "category", "description"]]
    categories = text_data["category"].unique()
    output_pc_h5 = h5py.File(Path(args.output) / "aligned_pc_data.hdf5", 'w')
    output_text_h5 = h5py.File(Path(args.output) / "aligned_text_data.hdf5", 'w')
    for category in categories:
        output_pc_h5.create_group(cate_to_synsetid[category.lower()])
        output_text_h5.create_group(cate_to_synsetid[category.lower()])
        category_text_data = text_data[text_data["category"] == category]
        for stage in ["test", "train", "val"]:
            category_stage_descriptions = []
            category_stage_pcs = None
            with open(Path(args.output) / f"shapenetcorev2_hdf5_2048/{stage:s}_files.txt".format(stage_t=stage), "r") as fileref:
                for file_path in fileref:
                    base_name = Path(file_path.strip()).stem
                    with open(Path(args.output) / f"shapenetcorev2_hdf5_2048/{base_name:s}_id2name.json", "r") as segment_fileref:
                        name_data = json.load(segment_fileref)
                    with open(Path(args.output) / f"shapenetcorev2_hdf5_2048/{base_name:s}_id2file.json", "r") as segment_fileref:
                        file_data = json.load(segment_fileref)
                    file_pc_ids = []
                    file_pc_id = -1
                    for model_category, model_file in zip(name_data, file_data):
                        file_pc_id += 1
                        if (model_category.lower() != category.lower()):
                            continue
                        model_id = Path(model_file).stem
                        model_descriptions = (category_text_data.loc[category_text_data["modelId"] == model_id])["description"].to_numpy(dtype = "str").tolist()
                        category_stage_descriptions += model_descriptions
                        file_pc_ids += [file_pc_id] * len(model_descriptions)
                    stage_pc_data = h5py.File(Path(args.output) / f"shapenetcorev2_hdf5_2048/{base_name:s}.h5", "r")
                    if (type(category_stage_pcs) == type(None)):
                        category_stage_pcs = stage_pc_data.get("data")[()][file_pc_ids]
                    else:
                        category_stage_pcs = np.append(category_stage_pcs, stage_pc_data.get("data")[()][file_pc_ids], axis = 0)
            print("_".join([category.lower(), stage, "data"]), ":", category_stage_pcs.shape)
            print("_".join([category.lower(), stage, "desc"]), ":", len(category_stage_descriptions))
            output_pc_h5[cate_to_synsetid[category.lower()]].create_dataset(stage, data = category_stage_pcs)
            output_text_h5[cate_to_synsetid[category.lower()]].create_dataset(stage, data = category_stage_descriptions)
