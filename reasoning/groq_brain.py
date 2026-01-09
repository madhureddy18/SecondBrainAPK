import os
import base64
from groq import Groq

# 🛡️ REPLACE WITH YOUR ACTUAL GROQ API KEY
# Get it from https://console.groq.com/keys
GROQ_API_KEY = "API_KEY"
client = Groq(api_key=GROQ_API_KEY)

def encode_image(image_path):
    """Helper to convert image to base64 for Groq Vision."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ask(text, lang="en", image_path=None):
    # System instruction for the blind user
    system_msg = (
        "You are an assistive 'Second Brain' for a blind person. "
        "Provide very concise, descriptive, and helpful answers. "
        f"Answer in {'Hindi' if lang == 'hi' else 'English'}."
    )

    try:
        if image_path and os.path.exists(image_path):
            # 🖼️ VISION MODE: Using Llama 4 Scout (Replacement for Llama 3.2 Vision)
            base64_image = encode_image(image_path)
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            # Updated Model ID
            model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            # 💬 TEXT MODE: Using Llama 3.3 70B
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text}
            ]
            model_name = "llama-3.3-70b-versatile"

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return (
            "I'm having trouble connecting to the Groq servers." 
            if lang == "en" else "मुझे ग्रोक सर्वर से जुड़ने में समस्या हो रही है।"
        )