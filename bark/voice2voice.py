from io import BytesIO

import torchaudio

import bark.utils
from bark import settings
from bark.core.api import semantic_to_waveform

from encodec.utils import convert_audio

from bark.model_downloader import get_hubert_manager_and_model
import numpy as np

def swap_voice_from_audio(
        audio_file: BytesIO | str,
        voice_name_or_embedding_path: str,
    ) -> tuple[np.ndarray, int]:
    """
    Takes voice and intonation from speaker_embedding and applies it to swap_audio_filename
    :param audio_file: the audio file to swap the voice. Can be a path or a file handle
    :param voice_name_or_embedding_path: the speaker embedding to use for the swap
    :return:
    """
    print("voice2voice")
    # Load the HuBERT model
    print("loading models")
    hubert_manager, hubert_model, model, tokenizer = get_hubert_manager_and_model()

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_file)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    device = bark.utils.get_cpu_or_gpu()
    wav = wav.to(device)

    # run inference
    print("inferencing")
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    audio = semantic_to_waveform(
        semantic_tokens,
        history_prompt=voice_name_or_embedding_path,
        temp=0.7,
        output_full=True
    )

    audio_array = audio.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    return audio_array, sample_rate


