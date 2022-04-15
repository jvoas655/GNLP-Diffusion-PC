from argparse import ArgumentParser
from pathlib import Path
from typing import List
import h5py
import numpy as np

from utilities.paths import DATA_FOLDER
from gnlp_diffusion_point_cloud.utils.dataset import cate_to_synsetid


if __name__ == "__main__":
    """
    Given a list of indices per text category (i.e. "chair" "table" etc.) - create a h5py dataset file exactly like
    the original aligned_data and aligned_text but limited strictly to the indices given. 
    
    This will not generated a small embeddings file (if you want cached embeddings for these new dataset files, run the
    cache embedding script on the newly created dataset files)
    """

    # Get CLI arguments
    parser = ArgumentParser()

    parser.add_argument('--indices', '-i', type=str, nargs='+', required=True,
                        help='List of indices separated by the category to be used in the new dataset.'
                             '(i.e. chair 1 5 19 39 table 49 210 230')
    parser.add_argument('--output_folder', '-o', type=str, default=str(DATA_FOLDER),
                        help='Location for the new data files')
    parser.add_argument('--output_file_prefix', '-p', type=str, default='small_data',
                        help='Multiple files will be created as output, this will prefix each file name.')
    parser.add_argument('--force', '-f', action='store_true', dest='force',
                        help='Overwrite output file if it exists')

    args = parser.parse_args()

    indices: List[str] = args.indices
    output_folder: Path = Path(args.output_folder)
    output_prefix: str = args.output_file_prefix
    force: bool = args.force

    # Parse the indices param and put them as a list associated with their sysetid key.
    category_indices = {}
    current_category = None
    for x in indices:
        if x in cate_to_synsetid:
            current_category = cate_to_synsetid[x]
        elif current_category:
            if current_category not in category_indices:
                category_indices[current_category] = []
            category_indices[current_category].append(int(x))

    # Set up output files, make sure they do not exist or user is okay overwriting them.
    output_pcs_file = output_folder / f'{output_prefix}_aligned_pc_data.hdf5'
    output_text_file = output_folder / f'{output_prefix}_aligned_text_data.hdf5'

    assert not output_pcs_file.exists() or force, f'File at {output_pcs_file} exists, use -f to overwrite'
    assert not output_text_file.exists() or force, f'File at {output_text_file} exists, use -f to overwrite'

    # Get the current data files.
    aligned_pc = DATA_FOLDER / 'aligned_pc_data.hdf5'
    aligned_text = DATA_FOLDER / 'aligned_text_data.hdf5'

    assert aligned_pc.exists(), f'Missing {aligned_pc} data file.'
    assert aligned_text.exists(), f'Missing {aligned_text} data file.'

    # Read in the data while creating dictionaries to temporarily store the new subset of the data.
    aligned_pc_data = h5py.File(aligned_pc)
    aligned_text_data = h5py.File(aligned_text)

    small_pcs = {}
    small_texts = {}

    for synsetid in list(category_indices.keys()):
        for idx in category_indices[synsetid]:
            # If it's not in one of the new files, create it for all of them.
            if synsetid not in small_pcs:
                small_pcs[synsetid] = {'train': []}
                small_texts[synsetid] = {'train': []}

            small_pcs[synsetid]['train'].append(aligned_pc_data[synsetid]['train'][idx])
            small_texts[synsetid]['train'].append(aligned_text_data[synsetid]['train'][idx])

        # Stack the list of numpy arrays
        small_pcs[synsetid]['train'] = np.stack(small_pcs[synsetid]['train'])
        small_texts[synsetid]['train'] = np.stack(small_texts[synsetid]['train'])

        # Make the train set the same for val and test (good for testing overfitting)
        small_pcs[synsetid]['val'] = small_pcs[synsetid]['train']
        small_pcs[synsetid]['test'] = small_pcs[synsetid]['train']

        small_texts[synsetid]['val'] = small_texts[synsetid]['train']
        small_texts[synsetid]['test'] = small_texts[synsetid]['train']

    # Write out the dataset files.
    small_datasets = [small_pcs, small_texts]
    output_files = [output_pcs_file, output_text_file]

    for ds, output in zip(small_datasets, output_files):
        with h5py.File(output, 'w') as f:
            for synsetid in list(ds.keys()):
                cate_group = f.create_group(synsetid)
                for split in list(ds[synsetid].keys()):
                    cate_group.create_dataset(split, data=ds[synsetid][split])

    print('Completed.  Output files created are:')
    print(f'{output_pcs_file}')
    print(f'{output_text_file}')
