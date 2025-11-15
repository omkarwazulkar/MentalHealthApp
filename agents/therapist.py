# agents/therapist.py

from autogen import AssistantAgent # type: ignore

def get_therapist(api_key):
    config = [{"model": "gpt-4.1-nano", "api_key": api_key}]
    return AssistantAgent(
        name="Therapist",
        llm_config={"config_list": config, "max_tokens": 100},
        system_message="You are a compassionate therapist. Use CBT techniques, validate feelings, and gently guide the user."
    )
