# testing/test_runners/voice_live_test.py
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import sounddevice as sd
import queue
import threading
import time

from voice.recognizer import VoskRecognizer
from voice.subtitle_engine import SubtitleEngine


# ------------------------------------------------------------
# Voice Live Test Runner
# ------------------------------------------------------------

def main():
    print("\n=== VOICE LIVE TEST ===\n")
    print("Initializing voice recognizer...")

    # --------------------------------------------------------
    # 1. Setup recognizer + subtitle engine
    # --------------------------------------------------------
    model_path = str(PROJECT_ROOT / "resources" / "vosk_model_small")
    samplerate = 16000

    recognizer = VoskRecognizer(model_path, samplerate=samplerate)
    subtitles = SubtitleEngine(silence_threshold=1.0, min_output_interval=0.4)

    audio_q = queue.Queue()

    # --------------------------------------------------------
    # 2. Audio callback
    # --------------------------------------------------------
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}")
        audio_q.put(bytes(indata))

    # --------------------------------------------------------
    # 3. Recognition loop (background thread)
    # --------------------------------------------------------
    def recognize_loop():
        while True:
            data = audio_q.get()

            if recognizer.accept_audio(data):
                final_text = recognizer.get_final()
                subtitles.process_final(final_text)
            else:
                partial = recognizer.get_partial()
                subtitles.process_partial(partial)

    threading.Thread(target=recognize_loop, daemon=True).start()

    # --------------------------------------------------------
    # 4. Start microphone stream
    # --------------------------------------------------------
    print("🎤 Starting microphone... (Ctrl+C to stop)\n")

    stream = sd.RawInputStream(
        samplerate=samplerate,
        blocksize=4000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    )
    stream.start()

    # --------------------------------------------------------
    # 5. Main loop: print subtitles
    # --------------------------------------------------------
    try:
        last_printed = ""
        while True:
            latest = subtitles.latest_lines(1)
            if latest:
                line = latest[0]
                if line != last_printed:
                    print(f"[VOICE] {line}")
                    last_printed = line
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Stopping voice test...")
    finally:
        stream.stop()
        stream.close()
        print("Done.\n")


if __name__ == "__main__":
    main()
