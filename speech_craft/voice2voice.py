from io import BytesIO

import torchaudio
from encodec.utils import convert_audio
import numpy as np

import speech_craft.supp.utils
from speech_craft.core.api import semantic_to_waveform
from speech_craft.settings import MODELS_DIR
from speech_craft.supp.model_downloader import get_hubert_manager_and_model, make_sure_models_are_downloaded


def voice2voice(
        audio_file: BytesIO | str,
        speaker_name_or_embedding_path: str,
    ) -> tuple[np.ndarray, int]:
    """
    Takes voice and intonation from speaker_embedding and applies it to swap_audio_filename
    :param audio_file: the audio file to swap the voice. Can be a path or a file handle
    :param speaker_name_or_embedding_path: the voice embedding to use for the swap
    :return:
    """
    #
    make_sure_models_are_downloaded(install_path=MODELS_DIR)
    # Load the HuBERT model
    hubert_manager, hubert_model, model, tokenizer = get_hubert_manager_and_model()

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_file)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    device = speech_craft.supp.utils.get_cpu_or_gpu()
    wav = wav.to(device)

    # run inference
    print("inferencing")
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    # move semantic tokens to cpu
    semantic_tokens = semantic_tokens.cpu().numpy()

    # convert voice2voice
    output_full = False
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=speaker_name_or_embedding_path,
        temp=0.7,
        output_full=output_full
    )
    if output_full:
        full_generation, audio_arr = out
    else:
        audio_arr = out

    return audio_arr, model.sample_rate


