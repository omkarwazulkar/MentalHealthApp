import speech_recognition as sr # type: ignore
import streamlit as st # type: ignore
def record_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"Voice recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
        return None