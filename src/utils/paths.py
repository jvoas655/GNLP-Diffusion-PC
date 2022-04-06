from pathlib import Path

SCRIPTS_FOLDER = Path(__file__).parent.resolve()
SRC_FOLDER = SCRIPTS_FOLDER.parent
ROOT_DIRECTORY = SRC_FOLDER.parent

DIFFUSION_MODEL_FOLDER = SRC_FOLDER / 'diffusion-point-cloud'
DIFFUSION_MODEL_DATA_FOLDER = DIFFUSION_MODEL_FOLDER / 'data'
DIFFUSION_MODEL_PRETRAINED_FOLDER = DIFFUSION_MODEL_FOLDER / 'pretrained'

