from openai import OpenAI # type: ignore
import streamlit as st # type: ignore

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
