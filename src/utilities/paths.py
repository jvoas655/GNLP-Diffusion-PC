from pathlib import Path

SCRIPTS_FOLDER = Path(__file__).parent.resolve()
SRC_FOLDER = SCRIPTS_FOLDER.parent
ROOT_DIRECTORY = SRC_FOLDER.parent

DIFFUSION_MODEL_FOLDER = SRC_FOLDER / 'diffusion-point-cloud'
GNLP_DIFFUSION_MODEL_FOLDER = SRC_FOLDER / 'gnlp-diffusion-point-cloud'
DATA_FOLDER = SRC_FOLDER / 'data'
PRETRAINED_FOLDER = SRC_FOLDER / 'pretrained'
RESULTS_FOLDER = SRC_FOLDER / 'results'
