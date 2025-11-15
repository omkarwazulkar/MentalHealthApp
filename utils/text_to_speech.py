from TTS.api import TTS # type: ignore
import tempfile
import soundfile as sf # type: ignore

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

# TTS
def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav = tts.tts(text=text)
        sf.write(tmp.name, wav, 22050)
        return tmp.name