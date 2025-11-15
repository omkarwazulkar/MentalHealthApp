import os
import time
import warnings
import streamlit as st # type: ignore
from agents.classify_specalist import classify_intent_with_gpt # type: ignore
from utils.text_to_speech import text_to_speech # type: ignore
from utils.record import record_voice # type: ignore
from utils.crisis import check_crisis # type: ignore
from models.emotion_model import detect_mood, ekman_to_sentiment # type: ignore
from agents.mindfulness import get_mindfulness # type: ignore
from agents.therapist import get_therapist # type: ignore
from agents.motivator import get_motivator # type: ignore
from agents.userproxy import get_proxy_agent # type: ignore
from pages.tab_03_journal import tab_journal
from pages.tab_04_mindfulness import tab_mindfulness

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

st.set_page_config(page_title="Multi-Agent Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Mental Health Chat")
api_key = st.secrets["OPENAI_API_KEY"]

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

            therapist = get_therapist(api_key)
            motivator = get_motivator(api_key)
            mindfulness = get_mindfulness(api_key)
            user_proxy = get_proxy_agent(api_key)

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
    tab_journal()

with tab4:
    tab_mindfulness()