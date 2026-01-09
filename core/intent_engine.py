def detect_intent(text):
    text = text.lower()

    # Keywords that trigger the vision system
    vision_keywords = [
        "see", "look", "what is", "describe", "camera", "vision",
        "objects", "items", "holding", "read", "how many",
        "मेरे सामने", "क्या है", "देखो", "आसपास", "कितने"
    ]

    for kw in vision_keywords:
        if kw in text:
            return "VISION"

    return "KNOWLEDGE"