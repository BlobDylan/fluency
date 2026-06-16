from pathlib import Path

ROOT_DIR = Path(__file__).parent
WEIGHTS_DIR = ROOT_DIR / "models" / "weights"

TOXICITY_MODEL_PATH = str(WEIGHTS_DIR / "toxic-bert")
FLUENCY_MODEL_PATH = str(WEIGHTS_DIR / "distilgpt2")