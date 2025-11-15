import torch # type: ignore
import streamlit as st # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore

model_path = "j-hartmann/emotion-english-distilroberta-base" # make sure it's the relative path
tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, token=True)

ekman_labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Neutral', 'Sadness', 'Surprise']
ekman_to_sentiment = {
    'Anger': 'Negative',
    'Disgust': 'Negative',
    'Fear': 'Negative',
    'Sadness': 'Negative',
    'Joy': 'Positive',
    'Surprise': 'Positive',
    'Neutral': 'Neutral'
}

def detect_mood(user_input):
    try:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        mood = ekman_labels[pred_id]
        return mood
    except Exception as e:
        st.error(f"Error in mood detection: {str(e)}")
        return "neutral"