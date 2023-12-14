import scipy
import torchaudio

from bark import settings
from bark.core.api import semantic_to_waveform

from encodec.utils import convert_audio

from bark.model_downloader import get_hubert_manager_and_model


def swap_voice_from_audio(
        swap_audio_filename: str,
        voice_embedding_path: str,
        out_path: str,
    ):
    """
    Takes voice and intonation from speaker_embedding and applies it to swap_audio_filename
    :param swap_audio_filename: the audio file where the swap is applied
    :param voice_embedding_path: the speaker embedding to use for the swap
    :return:
    """
    # Load the HuBERT model
    hubert_manager, hubert_model, model, tokenizer = get_hubert_manager_and_model()

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(swap_audio_filename)
    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)

    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    device = settings.get_cpu_or_gpu()
    wav = wav.to(device)

    # run inference
    semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    audio = semantic_to_waveform(
        semantic_tokens,
        history_prompt=voice_embedding_path,
        temp=0.7,
        output_full=True
    )

    audio_array = audio.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate

    scipy.io.wavfile.write(out_path, rate=sample_rate, data=audio_array)


