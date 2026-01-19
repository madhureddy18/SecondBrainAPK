import os
import base64
from groq import Groq

# 🛡️ REPLACE WITH YOUR ACTUAL GROQ API KEY
GROQ_API_KEY = "GROQ_API_KEY"
client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    """Helper to convert image to base64 for Groq Vision."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask(text, lang="en", image_path=None):
    # Map language codes to their full names for the AI instruction
    lang_map = {
        "hi": "Hindi",
        "te": "Telugu",
        "en": "English"
    }
    target_lang = lang_map.get(lang, "English")

    # CRITICAL: Updated instructions to explicitly support Telugu as per project goals
    system_msg = (
        "You are an assistive 'Second Brain' for a blind person. "
        "RULE 1: Always prioritize answering the user's text query directly. "
        "RULE 2: Use the provided image ONLY as context to answer that specific query. "
        "RULE 3: Do NOT start with 'The image shows' or 'I see' unless specifically asked to describe. "
        "RULE 4: If the image is unrelated to the query, answer the query using your general knowledge. "
        f"Answer concisely ONLY in {target_lang}. Do not provide English translations."
    )

    try:
        if image_path and os.path.exists(image_path):
            # Using the latest Llama 4 Scout for high-precision reasoning
            model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
            base64_image = encode_image(image_path)
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        # We place the text query FIRST to emphasize its importance
                        {"type": "text", "text": f"User's Question: {text}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        else:
            model_name = "llama-3.3-70b-versatile"
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text}
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1, # Keep temperature low for factual accuracy
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        # Multi-language error handling
        error_msgs = {
            "en": "I encountered a processing error.",
            "hi": "प्रसंस्करण में त्रुटि हुई।",
            "te": "ప్రాసెసింగ్‌లో లోపం ఏర్పడింది."
        }
        return error_msgs.get(lang, error_msgs["en"])