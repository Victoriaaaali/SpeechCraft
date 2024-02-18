import os

ROOT_DIR = os.getenv('ROOT_DIR', os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DEFAULT_SPEAKER_DIR = os.path.join(ROOT_DIR, "bark", "assets", "prompts")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

USE_GPU = os.getenv('USE_GPU', "True").lower() not in ('false', 'f', '0', 'off', 'n', 'no')
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(CUR_PATH, "assets", "prompts"))


def get_cpu_or_gpu() -> str:
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def set_embeddings_dir(path):
    global EMBEDDINGS_DIR
    EMBEDDINGS_DIR = path

def get_embeddings_dir():
    global EMBEDDINGS_DIR
    return EMBEDDINGS_DIR
