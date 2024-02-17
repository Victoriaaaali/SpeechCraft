import os

ROOT_DIR = os.environ.get('ROOT_DIR', os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_SPEAKER_DIR = os.path.join(ROOT_DIR, "bark", "assets", "prompts")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

DEFAULT_PORT = 8009
USE_GPU = os.environ.get('USE_GPU', "True").lower() not in ('false', 'f', '0', 'off', 'n', 'no')

