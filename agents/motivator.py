# agents/motivator.py

from autogen import AssistantAgent # type: ignore

def get_motivator(api_key):
    config = [{"model": "gpt-4.1-nano", "api_key": api_key}]
    return AssistantAgent(
        name="Motivator",
        llm_config={"config_list": config, "max_tokens": 100},
        system_message="You are a motivational coach. Offer encouragement, energy, clarity, and hype."
    )
