from tqdm import tqdm

import numpy as np
from bark.core.generation import SAMPLE_RATE, codec_decode, generate_coarse, generate_fine, generate_text_semantic
from bark.model_downloader import make_sure_models_are_downloaded
from bark.settings import MODELS_DIR

from io import BytesIO
from scipy.io import wavfile

from bark.utils import split_and_recombine_text


def text2voice_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):
    """
    :param text_prompt:
    :param semantic_temp:
    :param semantic_top_k:
    :param semantic_top_p:
    :param coarse_temp:
    :param coarse_top_k:
    :param coarse_top_p:
    :param fine_temp:
    :param voice_name:
    :param use_semantic_history_prompt:
    :param use_coarse_history_prompt:
    :param use_fine_history_prompt:
    :param output_full:
    :return:
    """

    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name if use_fine_history_prompt else None,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen,
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)


def text2voice(
        text: str,
        voice_name_or_embedding_path: str,
        temp_outfile_path: str = None,
        semantic_temp=0.7,
        semantic_top_k=50,
        semantic_top_p=0.95,
        coarse_temp=0.7,
        coarse_top_k=50,
        coarse_top_p=0.95,
        fine_temp=0.5
) -> (BytesIO, int):
    """
    :param text:
    :param voice_name_or_embedding_path:
    :param temp_outfile_path: If not none, the partial audio files will be stored in that directory
    :param semantic_temp:
    :param semantic_top_k:
    :param semantic_top_p:
    :param coarse_temp:
    :param coarse_top_k:
    :param coarse_top_p:
    :param fine_temp:
    :return:
    """

    make_sure_models_are_downloaded(install_path=MODELS_DIR)

    #texts = split_and_recombine_text(text)

    full_generation, audio_array = text2voice_with_settings(
        text,
        semantic_temp=semantic_temp,
        semantic_top_k=semantic_top_k,
        semantic_top_p=semantic_top_p,
        coarse_temp=coarse_temp,
        coarse_top_k=coarse_top_k,
        coarse_top_p=coarse_top_p,
        fine_temp=fine_temp,
        voice_name=voice_name_or_embedding_path,
        use_semantic_history_prompt=True,
        use_coarse_history_prompt=True,
        use_fine_history_prompt=True,
        output_full=True
    )

    # using virtual file to be able to return it as response instead of saving it
    wf = BytesIO()  # StringIO()
    wavfile.write(wf, SAMPLE_RATE, audio_array)

    return wf, SAMPLE_RATE