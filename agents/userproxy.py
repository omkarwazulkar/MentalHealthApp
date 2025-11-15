# agents/therapist.py

from autogen import UserProxyAgent # type: ignore

def get_proxy_agent(api_key):
    config = [{"model": "gpt-4.1-nano", "api_key": api_key}]
    return UserProxyAgent(
        name="User", 
        human_input_mode="NEVER", 
        code_execution_config={"use_docker": False}
    )
