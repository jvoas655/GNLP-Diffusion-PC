from argparse import ArgumentParser
from pathlib import Path
import gdown
import zipfile
import os

from utilities.paths import DATA_FOLDER


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('-o', '--output', type=str,
                           help='Shapenet File location, default is (Diffusion Models Directory under data/)',
                           default=str(DATA_FOLDER))
    argparser.add_argument('-f', '--force', action='store_true', dest='force',
                           help='Overwrite anything that may exist in the current output path.')
    argparser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                           help='Quiet flag passed into gdown.')

    args = argparser.parse_args()

    output_path: Path = Path(args.output) / 'shapenet.hdf5'

    force: bool = args.force
    quiet: bool = args.quiet

    assert not output_path.exists() or force, f"Something already exists at {output_path}, pick a new place or use -f."

    data_file_id = '1WdooIyOMVt_HVd_A0QOM9DTTX41O65jQ'
    gdown.download(output=str(output_path), quiet=quiet, id=data_file_id)
