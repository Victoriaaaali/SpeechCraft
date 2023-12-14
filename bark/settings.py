import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(ROOT_DIR), "models")
DEFAULT_SPEAKER_DIR = os.path.join(os.path.dirname(ROOT_DIR), "bark", "assets", "prompts")

DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "output")
DEFAULT_PORT = 8009

def get_cpu_or_gpu() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

