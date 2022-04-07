from argparse import ArgumentParser
from pathlib import Path
import gdown
import zipfile
import os

from utils.paths import DIFFUSION_MODEL_DATA_FOLDER


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('-o', '--output', type=str,
                           help='Shapenet File location, default is (Diffusion Models Directory under data/)',
                           default=str(DIFFUSION_MODEL_DATA_FOLDER))
    argparser.add_argument('-f', '--force', action='store_true', dest='force',
                           help='Overwrite anything that may exist in the current output path.')
    argparser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                           help='Quiet flag passed into gdown.')

    args = argparser.parse_args()

    output_path: Path = Path(args.output) / 'aligned_text_pc.zip'
    force: bool = args.force
    quiet: bool = args.quiet

    assert not output_path.exists() or force, f"Something already exists at {output_path}, pick a new place or use -f."

    data_file_id = '1S8Gzf5HX5MvN0VyVaNd1Sp3H1lVN_NKr'
    gdown.download(output=str(output_path), quiet=quiet, id=data_file_id)

    with zipfile.ZipFile(Path(args.output) / 'aligned_text_pc.zip', 'r') as zip_ref:
        zip_ref.extractall(Path(args.output))
    os.remove(Path(args.output) / 'aligned_text_pc.zip')
