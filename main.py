import time
import sys
from core.state_manager import StateManager
from core.intent_engine import detect_intent
from core.memory import Memory

from perception.audio_input import record_audio
from perception.speech_to_text import transcribe
from perception.vision import get_vision_data # Updated import

from reasoning.groq_brain import ask
from utils.language import detect_language, is_valid_speech
from output.tts import speak
from utils.sounds import beep

# -------------------------
# INITIALIZE SYSTEM
# -------------------------
state = StateManager()
memory = Memory()

print("Second Brain starting...")

# Startup cue
beep(1000, 500)
speak("Hi, how can I help you?", "en")

def run_interaction_cycle():
    # 1. LISTENING STATE
    state.set_state(StateManager.LISTENING)
    print("\n[SYSTEM] Listening...")
    beep(800, 150) 
    record_audio("input.wav", duration=5)

    # 2. PROCESSING STATE
    state.set_state(StateManager.PROCESSING)
    text = transcribe("input.wav")
    print(f"User: {text}")

    # Check for exit commands
    if text.lower().strip() in ["stop", "shutdown", "exit", "बंद करो"]:
        speak("Shutting down the second brain. Goodbye.", "en")
        sys.exit()

    if not is_valid_speech(text):
        if len(text.strip()) > 0:
            speak("I didn't quite catch that. Could you repeat?", "en")
        return

    lang = detect_language(text)
    intent = detect_intent(text)

    # 3. RESPONDING STATE
    state.set_state(StateManager.RESPONDING)
    
    if intent == "VISION":
        print("[PROCESS] Analyzing environment...")
        # Captures image and local object counts
        detections, image_path = get_vision_data()
        
        if image_path is None:
            response = "The camera is unavailable." if lang == "en" else "कैमरा उपलब्ध नहीं है।"
        else:
            # We pass the user's question and the image to the Groq Vision Brain.
            # This identifies phones, bottles, fruits, and people naturally.
            response = ask(text, lang, image_path=image_path)
    else:
        # General Conversation / AI Reasoning
        response = ask(text, lang)

    # LOUD AUDIO OUTPUT
    print(f"Brain: {response}")
    speak(response, lang)

    # 4. RESET TO IDLE
    state.set_state(StateManager.IDLE)
    time.sleep(1.5) 

if __name__ == "__main__":
    while True:
        try:
            run_interaction_cycle()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Critical System Error: {e}")
            time.sleep(2)