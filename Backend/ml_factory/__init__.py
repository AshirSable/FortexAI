from pathlib import Path
import os

BASE_DIR = Path(os.path.dirname(__file__))
MODELS_DIR = BASE_DIR / 'models'
MODEL_DIRECTORY_DIR = BASE_DIR / 'model_directory'
TRAINING_SCRIPT = BASE_DIR / 'training_scripts'
DATA_DIR = BASE_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
TRAINING_LOGS_DIR = BASE_DIR / 'training_logs'
MODEL_DIRECTORY_RELEASED = MODEL_DIRECTORY_DIR / 'released'
MODEL_DIRECTORY_DEV = MODEL_DIRECTORY_DIR / 'development'

MODEL_DIRECTORY_DIR.mkdir(exist_ok=True)
MODEL_DIRECTORY_RELEASED.mkdir(exist_ok=True)
MODEL_DIRECTORY_DEV.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
DATA_RAW_DIR.mkdir(exist_ok=True)
DATA_PROCESSED_DIR.mkdir(exist_ok=True)
