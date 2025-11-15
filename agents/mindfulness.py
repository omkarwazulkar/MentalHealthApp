# agents/mindfulness.py

from autogen import AssistantAgent # type: ignore

def get_mindfulness(api_key):
    config = [{"model": "gpt-4.1-nano", "api_key": api_key}]
    return AssistantAgent(
        name="MindfulnessCoach",
        llm_config={"config_list": config, "max_tokens": 100},
        system_message="You are a mindfulness coach. Teach breathing, grounding, awareness, and calm."
    )
