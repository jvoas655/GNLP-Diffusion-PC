from argparse import ArgumentParser
from pathlib import Path
import gdown

from utils.paths import DIFFUSION_MODEL_PRETRAINED_FOLDER


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument('-o', '--output', type=str,
                           help='Folder path to put all the pretrained models.',
                           default=str(DIFFUSION_MODEL_PRETRAINED_FOLDER))
    argparser.add_argument('-f', '--force', action='store_true', dest='force',
                           help='Overwrite anything that may exist in the current output path.')
    argparser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                           help='Quiet flag passed into gdown.')

    args = argparser.parse_args()

    output_path: Path = Path(args.output)
    force: bool = args.force
    quiet: bool = args.quiet

    pretrained_files_name_and_id = [
        ('AE_airplane.pt', '10QcOQzfszy053WQJEzdSNgytdiPin3nU'),
        ('AE_all.pt', '1Bku4qfFaWqm80NqF1A0_KaepVdxY9Q_l'),
        ('AE_car.pt', '1R8UBQgyUKwBn7WuYuvYpuMacH5kRWaHh'),
        ('AE_chair.pt', '1pOq4AMc6auJAWVlG0ei4d9vbGe6h8dJL'),
        ('GEN_airplane.pt', '1NTVl1RKcIJDildrTx-jUpBJkDNzg55ld'),
        ('GEN_chair.pt', '1qXtJCOevgdyhxwllJ5G4e7ZGH0sFsDXF')
    ]

    # Quick check to make sure nothing is already in those file locations.
    for name, file_id in pretrained_files_name_and_id:
        file_output = output_path / name
        assert not file_output.exists() or force, f"Something already exists at {file_output}, change loc or use -f."

    for name, file_id in pretrained_files_name_and_id:
        file_output = output_path / name
        gdown.download(output=str(file_output), quiet=quiet, id=file_id)