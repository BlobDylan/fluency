from pathlib import Path

ROOT_DIR = Path(__file__).parent
WEIGHTS_DIR = ROOT_DIR / "models" / "weights"

TOXICITY_MODEL_PATH = str(WEIGHTS_DIR / "toxic-bert")
FLUENCY_MODEL_PATH = str(WEIGHTS_DIR / "distilgpt2")
TARGET_MODEL_PATH = str(WEIGHTS_DIR / "llama-3.2-3b-instruct")
AGENT_MODEL_PATH = str(WEIGHTS_DIR / "qwen2.5-0.5b")