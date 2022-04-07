from argparse import ArgumentParser
from pathlib import Path
import gdown
import h5py
import pandas as pd
import os
import json
import numpy as np

from utils.paths import DIFFUSION_MODEL_DATA_FOLDER

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('-o', '--output', type=str,
                           help='Shapenet File location, default is (Diffusion Models Directory under data/)',
                           default=str(DIFFUSION_MODEL_DATA_FOLDER))

    args = argparser.parse_args()

    text_data = pd.read_csv(Path(args.output) / "text2shape.captions.csv")[["modelId", "category", "description"]]
    categories = text_data["category"].unique()
    output_h5 = h5py.File(Path(args.output) / "aligned_text_pc.hdf5", 'w')
    for category in categories:
        category_group = output_h5.create_group(category)
        category_text_data = text_data[text_data["category"] == category]
        for stage in ["test", "train", "val"]:
            category_stage_descriptions = []
            category_stage_pcs = None
            with open(Path(args.output) / "shapenetcorev2_hdf5_2048\{stage_t:s}_files.txt".format(stage_t=stage), "r") as fileref:
                for file_path in fileref:
                    base_name = Path(file_path.strip()).stem
                    with open(Path(args.output) / "shapenetcorev2_hdf5_2048\{base:s}_id2name.json".format(base=base_name), "r") as segment_fileref:
                        name_data = json.load(segment_fileref)
                    with open(Path(args.output) / "shapenetcorev2_hdf5_2048\{base:s}_id2file.json".format(base=base_name), "r") as segment_fileref:
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
                    stage_pc_data = h5py.File(Path(args.output) / "shapenetcorev2_hdf5_2048\{base:s}.h5".format(base=base_name), "r")
                    if (type(category_stage_pcs) == type(None)):
                        category_stage_pcs = stage_pc_data.get("data")[()][file_pc_ids]
                    else:
                        category_stage_pcs = np.append(category_stage_pcs, stage_pc_data.get("data")[()][file_pc_ids], axis = 0)
            print("_".join([category.lower(), stage, "data"]), ":", category_stage_pcs.shape)
            print("_".join([category.lower(), stage, "desc"]), ":", len(category_stage_descriptions))
            category_group.create_dataset("_".join([category.lower(), stage, "data"]), data = category_stage_pcs)
            category_group.create_dataset("_".join([category.lower(), stage, "desc"]), data = category_stage_descriptions)
