from text2voice import text2voice, voice2voice, voice2embedding
from media_toolkit import AudioFile

sample_text = "I love socAIty [laughs]! [happy] What a day to make voice overs."

# test text2speech
#audio_numpy, sample_rate = text2voice(sample_text)
#audio = AudioFile().from_np_array(audio_numpy, sr=sample_rate)
#audio.save("test_audio.wav")
#
## test voice cloning
#embedding = voice2embedding(audio_file="test_files/voice_clone_test_voice_1.wav", speaker_name="hermine").save_to_speaker_lib()
#tts_new_speaker, sample_rate = text2voice(sample_text, voice=embedding)
#audio_with_cloned_voice = AudioFile().from_np_array(tts_new_speaker, sr=sample_rate)
#audio_with_cloned_voice.save("test_audio_cloned.wav")

# test voice2voice
cloned_audio = voice2voice(audio_file="test_files/voice_clone_test_voice_2.wav", speaker_name_or_embedding_path="hermine")

a =1