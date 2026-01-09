import edge_tts
import asyncio
import pygame
import os
import time

def speak(text, lang="en"):
    """High-quality neural TTS that supports English and Hindi."""
    if not text or len(text.strip()) == 0:
        return

    # Map languages to Neural Voices
    voice = "hi-IN-MadhurNeural" if lang == "hi" else "en-US-GuyNeural"
    output_file = "temp_voice.mp3"

    async def _generate():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_file)

    try:
        # 1. Generate the audio file
        asyncio.run(_generate())

        # 2. Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        # 3. Wait for audio to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        # 4. Clean up temporary file
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass