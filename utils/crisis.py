def check_crisis(text):
    crisis_keywords = ["suicide", "kill myself", "end my life", "can't go on", "self-harm"]
    return any(word in text.lower() for word in crisis_keywords)
