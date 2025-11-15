import requests # type: ignore
from gtts import gTTS # type: ignore
import tempfile
import streamlit as st # type: ignore

def speak(text):
    tts = gTTS(text)
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(f.name)
    return f.name