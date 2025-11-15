import requests # type: ignore
import tempfile
import streamlit as st # type: ignore

def hf_tts(text: str):
    hf_token = st.secrets["HF_TOKEN"]

    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"

    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {"inputs": text}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace TTS failed: {response.text}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(response.content)
        return tmp.name