from pathlib import Path

SCRIPTS_FOLDER: Path = Path(__file__).parent.resolve()
SRC_FOLDER: Path = SCRIPTS_FOLDER.parent
ROOT_DIRECTORY: Path = SRC_FOLDER.parent

DIFFUSION_MODEL_FOLDER: Path = SRC_FOLDER / 'diffusion_point_cloud'
GNLP_DIFFUSION_MODEL_FOLDER: Path = SRC_FOLDER / 'gnlp_diffusion_point_cloud'
DATA_FOLDER: Path = SRC_FOLDER / 'data'
PRETRAINED_FOLDER: Path = SRC_FOLDER / 'pretrained'
RESULTS_FOLDER: Path = SRC_FOLDER / 'results'
