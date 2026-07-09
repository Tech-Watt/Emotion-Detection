from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "data.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
