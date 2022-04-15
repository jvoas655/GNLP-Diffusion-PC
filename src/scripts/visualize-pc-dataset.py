import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List
import h5py

from utilities.paths import DATA_FOLDER
from utilities.visualizations import draw_pc
from gnlp_diffusion_point_cloud.utils.dataset import cate_to_synsetid, synsetid_to_cate


def command_help():
    print("""
    h: help
    
    n: next point cloud in the index list of the current category
    '' or enter: same as 'n'
    p: previous point cloud in the index list of the current category
    
    nc: next category
    pc: previous category
    
    e: exit
    """)


if __name__ == "__main__":
    """
    Given a path to a dataset file that holds point clouds generate scatter plots of each PC 
    """

    parser = ArgumentParser()

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='The path to an HDF5 file that contains point cloud [N, K, 3] data.'
                             'if you do not include / the DATA_FOLDER is assumed')
    parser.add_argument('--categories', '-c', type=str, nargs='+', default=(),
                        help='Categories you want to visualize (ex. table chair bus)')
    parser.add_argument('--split', '-s', type=str, choices=['train', 'val', 'test'], default='train',
                        help='Data split to view.  Options are train, val, and test.')
    parser.add_argument('--indices', '-idx', type=str, nargs='+', default=(),
                        help='Indices to visualize from each category. Separate disjoint indices, use : for ranges '
                             '(i.e. 1 2 3 10:20 50 100')

    args = parser.parse_args()

    input_file: Path = Path(args.input) if '/' in args.input else DATA_FOLDER / args.input
    categories: List[str] = args.categories
    split: str = args.split
    string_indices = args.indices

    assert input_file.exists(), f'{input_file} does not exist!'

    indices: List[int] = []
    for idx in string_indices:
        if ':' in idx:
            start = int(idx.split(':')[0])
            end = int(idx.split(':')[1])
            [indices.append(x) for x in range(start, end)]
        else:
            indices.append(int(idx))

    data = h5py.File(str(input_file))

    if len(categories) == 0:
        categories = list(data.keys())
    else:
        categories = [cate_to_synsetid(x) for x in categories]

    # Show plots
    c_idx = 0
    n_idx = 0
    while True:
        category = categories[c_idx]
        category_indices = []
        if len(indices) == 0:
            category_indices = list(range(len(data[category][split])))

        draw_pc(
            data[category][split][category_indices[n_idx]],
            f'{synsetid_to_cate[category]}: {category_indices[n_idx]}'
        )

        cmd = None
        while not cmd or cmd == 'h':
            cmd = input("Command: ")

            if cmd == 'h':
                command_help()
            elif cmd == '':
                break

        # Really stupid code to keep you in bounds of the dataset indices.
        if cmd == 'e':
            print("Exiting")
            sys.exit(0)
        elif cmd == 'n' or cmd == '':
            if n_idx >= len(category_indices) - 1:
                if c_idx >= len(categories) - 1:
                    continue
                c_idx += 1
                n_idx = 0
                continue
            n_idx += 1
        elif cmd == 'p':
            if n_idx == 0:
                if c_idx == 0:
                    continue
                c_idx -= 1
                n_idx = 0
                continue
            n_idx -= 1
        elif cmd == 'nc':
            if c_idx >= len(categories) - 1:
                continue
            c_idx += 1
            n_idx = 0
        elif cmd == 'pc':
            if c_idx == 0:
                continue
            c_idx -= 1
            n_idx = 0
