import time
import sys
from core.state_manager import StateManager
from core.intent_engine import detect_intent
from core.memory import Memory

from perception.audio_input import record_audio
from perception.speech_to_text import transcribe
from perception.vision import get_vision_data 

from reasoning.groq_brain import ask
from utils.language import detect_language, is_valid_speech
from output.tts import speak
from utils.sounds import beep

state = StateManager()
memory = Memory()

print("Second Brain starting...")
beep(1000, 500)
speak("Hi, how can I help you?", "en")

def run_interaction_cycle():
    state.set_state(StateManager.LISTENING)
    print("\n[SYSTEM] Listening...")
    beep(800, 150) 
    record_audio("input.wav", duration=5)

    state.set_state(StateManager.PROCESSING)
    text = transcribe("input.wav")
    print(f"User: {text}")

    if text.lower().strip() in ["stop", "shutdown", "exit", "बंद करो"]:
        speak("Shutting down the second brain. Goodbye.", "en")
        sys.exit()

    if not is_valid_speech(text):
        if len(text.strip()) > 0:
            speak("I didn't quite catch that. Could you repeat?", "en")
        return

    lang = detect_language(text)
    intent = detect_intent(text)

    state.set_state(StateManager.RESPONDING)
    
    if intent == "VISION":
        # Robustness: Notify the user so they hold the camera still
        speak("Analyzing environment, please hold steady.", lang)
        print("[PROCESS] Capturing stabilized frame...")
        
        detections, image_path = get_vision_data()
        
        if image_path is None:
            response = "The camera is unavailable." if lang == "en" else "कैमरा उपलब्ध नहीं है।"
        else:
            # Using the enhanced vision brain for high-accuracy reasoning
            response = ask(text, lang, image_path=image_path)
    else:
        response = ask(text, lang)

    print(f"Brain: {response}")
    speak(response, lang)

    state.set_state(StateManager.IDLE)
    time.sleep(1.0) 

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