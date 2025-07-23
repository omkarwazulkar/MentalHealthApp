import os
import time
import tempfile
import torch # type: ignore
from gtts import gTTS # type: ignore
import streamlit as st # type: ignore
from openai import OpenAI # type: ignore
from textblob import TextBlob # type: ignore
import speech_recognition as sr # type: ignore
from autogen import AssistantAgent, UserProxyAgent # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification # type: ignore

st.set_page_config(page_title="Multi-Agent Mental Health Chatbot", page_icon="🧠")

# Title
st.title("🧠 Multi-Agent Mental Health Chatbot")

# API Key Input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_history" not in st.session_state:
    st.session_state.mood_history = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {"name": None, "goals": [], "preferences": {}}

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Voice Input ---------------- #
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

# ---------------- TTS Output ---------------- #
def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name


# ---------------- Load Model ---------------- #
model_path = "ekman_emotion_model"  # make sure it's the relative path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label maps
ekman_labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Neutral']
ekman_to_sentiment = {
    'Anger': 'Negative',
    'Disgust': 'Negative',
    'Fear': 'Negative',
    'Sadness': 'Negative',
    'Joy': 'Positive',
    'Surprise': 'Positive',
    'Neutral': 'Neutral'
}

# Mood detection using fine-tuned model
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

# ---------------- Crisis Detection ---------------- #
def check_crisis(text):
    crisis_keywords = ["suicide", "kill myself", "end my life", "can't go on", "self-harm"]
    return any(word in text.lower() for word in crisis_keywords)

# ---------------- Classify Intent ---------------- #
def classify_intent_with_gpt(user_input, mood, api_key):
    client = OpenAI(api_key=api_key)

    routing_prompt = f"""
You are an intelligent assistant that routes mental health user messages to the most appropriate virtual support agent.

User input: "{user_input}"
Detected mood: {mood}

Based on this, decide which agent should respond:

1. Therapist → If the user expresses emotional pain, depression, trauma, sadness, confusion, fear, or irrational thinking.
2. Motivator → If the user expresses lack of energy, procrastination, self-doubt, laziness, or need encouragement.
3. MindfulnessCoach → If the user expresses stress, anxiety, a racing mind, need for calmness, grounding, or mindfulness exercises.

Respond ONLY with one word:
Therapist, Motivator, or MindfulnessCoach.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": routing_prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT Intent Classification failed: {str(e)}")
        return "Therapist"  # fallback if GPT fails

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Mood Tracker", "📔 Journal", "🧘 Mindfulness"])

with tab1:
    st.subheader("Your Conversation")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.chat_input("Type your message or use voice...")
    with col2:
        if st.button("🎤 Speak"):
            voice_text = record_voice()
            if voice_text:
                user_input = voice_text

    if user_input:
        mood = detect_mood(user_input)
        selected_agent_name = classify_intent_with_gpt(user_input, mood, api_key)
        sentiment = ekman_to_sentiment[mood]
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.mood_history.append((time.strftime("%Y-%m-%d %H:%M"), mood))

        with st.chat_message("user"):
            st.markdown(f"**{user_input}**\n\nMood: {mood} \t\t\t Sentiment: {sentiment}")

        # Crisis Check
        if check_crisis(user_input):
            st.error("⚠️ It sounds like you're in crisis. Please reach out to emergency services or call a hotline:\n\n**Suicide Hotline:** 988 (US) | **UK:** 0800 689 5652")
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            config_list = [{"model": "gpt-4.1-nano", "api_key": api_key}]

            # Define agents
            therapist = AssistantAgent(name="Therapist", llm_config={"config_list": config_list, "max_tokens":100},
                                       system_message="You are a compassionate therapist. Use CBT and empathy.")
            motivator = AssistantAgent(name="Motivator", llm_config={"config_list": config_list, "max_tokens":100},
                                       system_message="You provide uplifting and encouraging messages.")
            mindfulness = AssistantAgent(name="MindfulnessCoach", llm_config={"config_list": config_list, "max_tokens":100},
                                         system_message="Suggest mindfulness and breathing exercises.")
            user_proxy = UserProxyAgent(name="User", human_input_mode="NEVER", code_execution_config={"use_docker": False})

            conversation = [
                {"role": "system", "content": f"User mood is {mood}. Provide combined advice."},
                {"role": "user", "content": user_input}
            ]

            # Multi-agent responses
            if selected_agent_name == "Therapist":
                reply = therapist.generate_reply(messages=conversation)
            elif selected_agent_name == "Motivator":
                reply = motivator.generate_reply(messages=conversation)
            elif selected_agent_name == "MindfulnessCoach":
                reply = mindfulness.generate_reply(messages=conversation)
            else:
                reply = therapist.generate_reply(messages=conversation)  # fallback

            # Format and show final response
            final_response = f"**{selected_agent_name}:** {reply}"

            # Add to session history for display
            st.session_state.messages.append({"role": "assistant", "content": final_response})

            # Show assistant message in chat UI
            with st.chat_message("assistant"):
                st.markdown(final_response)

            # Audio response
            audio_file = text_to_speech(final_response)
            st.audio(audio_file, format="audio/mp3")
        else:
            st.warning("Please enter your API key!")

with tab2:
    st.subheader("Mood Trend")
    if st.session_state.mood_history:
        st.line_chart({"Mood": [m[1] for m in st.session_state.mood_history]})
    else:
        st.write("No mood data yet.")

with tab3:
    st.subheader("Your Journal")
    journal_text = st.text_area("Write your thoughts for today:")
    if st.button("Save Journal Entry"):
        st.success("Journal saved!")

with tab4:
    st.subheader("Mindfulness Exercise")
    st.write("Try the 4-7-8 breathing technique:\n- Inhale for 4 seconds\n- Hold for 7 seconds\n- Exhale for 8 seconds")
